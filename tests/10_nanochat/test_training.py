"""Section 10.3 — entraînement de contrôle : le modèle nanochat-like apprend-il par cœur ?

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/10_nanochat/test_training.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(100 pas, ln(32) ≈ 3.4657 comme perte initiale, 0.5 comme plafond final) : c'est volontaire.
Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code
testé.

Modèle jouet de la section 10, constant dans tout le chapitre : `vocab_size=32`,
`hidden_size=16`, `num_layers=2`, `num_heads=4`, `num_kv_heads=2`, `head_dim=4`,
`intermediate_size=32`, `seq=6`, `batch=1`. Ce fichier rejoue sur l'architecture moderne le
smoke test d'entraînement de la section 3.11, et réutilise volontairement la MÊME API
`train_steps` : la boucle d'entraînement ne dépend pas de l'architecture, seul le modèle
change. C'est aussi le seul contrôle du chapitre qui traverse RMSNorm, RoPE, GQA et SwiGLU
en marche arrière : un `epsilon` de norme mal placé, une rotation RoPE appliquée à V ou un
`repeat_interleave` de têtes KV dans le mauvais sens cassent le gradient bien avant de casser
une shape.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
@pytest.mark.model
def test_nanochat_like_model_can_overfit_tiny_batch():
    """Roadmap 10.3 — si l'architecture complète n'apprend pas UN batch, elle est cassée.

    Objectif d'apprentissage
    ------------------------
    « Overfit a single batch » est le test de fumée le plus rentable du deep learning : un
    modèle de quelques milliers de paramètres doit pouvoir apprendre par cœur six tokens. S'il
    n'y arrive pas, le problème n'est jamais le jeu de données, c'est le code — graphe
    d'autograd coupé, masque causal inversé, normalisation qui écrase le signal, têtes KV
    dupliquées dans le mauvais ordre.

    Le deuxième enseignement est la valeur de DÉPART. Un modèle fraîchement initialisé ne sait
    rien : sa distribution est quasi uniforme sur les 32 tokens, donc son entropie croisée
    vaut environ `ln(32) ≈ 3.4657`. Ce chiffre est un outil de diagnostic redoutable, et il
    contraint l'initialisation : avec une table d'embedding tirée en `N(0, 1)` et le weight
    tying (`lm_head.weight is embedding.weight`), les logits initiaux valent plusieurs
    dizaines et la perte de départ dépasse 15 — le modèle démarre avec des convictions
    aberrantes. L'initialisation `N(0, 0.02)` de nanochat et GPT-2 est précisément ce qui
    ramène la perte initiale sur `ln(vocab)`.

    Schéma mental
    -------------
        données : token_ids (batch=1, seq=6) et targets (1, 6) = ids décalés d'une position
        modèle  : vocab=32, hidden=16, num_layers=2, num_heads=4, num_kv_heads=2   (graine 0)

        pas 0  : cross_entropy(logits.reshape(6, 32), targets.reshape(6)) ~= ln(32) = 3.4657
        ...      100 pas de descente de gradient sur CE SEUL exemple
        pas 99 : perte très basse (< 0.5), le modèle a appris par cœur

        losses = [~3.47, ..., ~0.0]        max(losses[-10:]) < min(losses[:10])

    Ce que ce test vérifie
    ----------------------
    1. la boucle rend l'historique complet des pertes : une valeur par pas, soit 100 valeurs,
       et 1 seule valeur si on ne demande qu'un pas (un historique, pas seulement la dernière
       perte : sinon on ne peut rien diagnostiquer) ;
    2. toutes les pertes sont des flottants finis et strictement positifs (une entropie croisée
       négative ou `nan` trahit un bug de perte ou de gradient) ;
    3. la perte initiale vaut environ `ln(32) ≈ 3.4657`, l'entropie d'une distribution uniforme
       sur 32 tokens : le modèle part bien de l'ignorance, donc son initialisation est saine ;
    4. la perte descend vraiment — fin de trajectoire strictement sous le début, et perte
       finale nettement sous la barre de l'uniforme (< 0.5, soit sept fois moins que 3.4657).

    API à faire émerger (cible roadmap « modèle/training », cible proposée :
    `src/inference_lab/training/loop.py`, module déjà visé par 3.11)
    -----------------------------------------------------------------------
        def train_steps(
            model: torch.nn.Module,
            token_ids: torch.Tensor,
            targets: torch.Tensor,
            steps: int,
            learning_rate: float,
        ) -> list[float]: ...

    C'est exactement la signature de 3.11 : la boucle d'entraînement est indépendante de
    l'architecture, elle ne voit du modèle qu'un appelable rendant des logits. Aucun code
    supplémentaire ne devrait être nécessaire ici si 3.11 est déjà au vert.

    Indice : `torch.optim.Adam(model.parameters(), lr=0.05)` sur 100 pas suffit largement pour
    ce modèle (avec un SGD nu, il faut beaucoup plus de pas et l'assert 4 devient capricieux).
    Pièges : oublier `optimizer.zero_grad()`, aplatir logits et cibles de façon incohérente
    (`cross_entropy` attend `(N, C)` et `(N,)`), et surtout initialiser les poids par défaut —
    l'assert 3 exige une initialisation `N(0, 0.02)`, notamment sur l'embedding lorsqu'il est
    partagé avec le `lm_head`.
    """

    pytest.skip("Roadmap TDD 10.3 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.training.loop import train_steps

    # Arrange — fixer la graine avec `torch.manual_seed(0)` AVANT toute création de tenseur,
    #           puis construire :
    #           - `model`, le `NanochatLikeModel` de la section 10.2 (vocab_size=32,
    #             hidden_size=16, num_layers=2, num_heads=4, num_kv_heads=2, head_dim=4,
    #             intermediate_size=32), en mode entraînement, avec des poids initialisés en
    #             loi normale d'écart-type 0.02 ;
    #           - `token_ids`, un tenseur `torch.long` de shape (batch=1, seq=6) d'ids dans
    #             [0, 32), et `targets`, de même shape, contenant la suite à prédire (les
    #             mêmes ids décalés d'une position) ;
    #           - `steps = 100` et `learning_rate = 0.05`, assez pour que ce modèle minuscule
    #             apprenne par cœur cet unique exemple.

    # Act — lancer la boucle d'entraînement sur ces données et récupérer `losses`,
    #       l'historique des pertes pas à pas.

    # Assert 1 — un historique complet, une perte par pas, quel que soit le nombre de pas
    assert isinstance(losses, list)
    assert len(losses) == 100
    assert len(train_steps(model, token_ids, targets, steps=1, learning_rate=0.05)) == 1

    # Assert 2 — des pertes finies et strictement positives
    assert all(isinstance(loss, float) for loss in losses)
    assert bool(torch.isfinite(torch.tensor(losses)).all())
    assert min(losses) > 0.0

    # Assert 3 — au départ, le modèle est ignorant : entropie de l'uniforme sur 32 tokens
    assert losses[0] == pytest.approx(3.4657, abs=0.5)

    # Assert 4 — la perte décroît et passe NETTEMENT sous la barre de l'uniforme
    assert losses[-1] < losses[0]
    assert max(losses[-10:]) < min(losses[:10])
    assert losses[-1] < 0.5
