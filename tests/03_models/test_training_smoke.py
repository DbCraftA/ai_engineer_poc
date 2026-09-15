"""Section 3.11 — smoke test d'entraînement : la boucle apprend-elle vraiment ?

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/03_models/test_training_smoke.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(50 pas, ln(16) ≈ 2.77 comme perte initiale) : c'est volontaire. Un test doit énoncer la
vérité attendue, pas la recalculer avec la même formule que le code testé.

Modèle jouet de la section 3, constant dans tout le fichier : `vocab_size=16`, `hidden=8`,
`num_layers=2`, `num_heads=2`, `seq=4`, `batch=1`. C'est le seul test de la roadmap qui
entraîne quelque chose : le reste du dépôt est consacré à l'inférence. Son rôle est de
prouver que le modèle assemblé en 3.3 est bien dérivable de bout en bout — un masque causal
inversé, une normalisation cassée ou un `detach()` malencontreux se voient immédiatement ici,
alors qu'ils passeraient inaperçus dans un simple contrôle de shapes.

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
def test_tiny_model_can_reduce_loss_on_tiny_dataset():
    """Roadmap 3.11 — si la perte ne descend pas sur un jeu minuscule, le modèle est cassé.

    Objectif d'apprentissage
    ------------------------
    Un modèle jouet doit pouvoir apprendre par cœur un jeu de données minuscule : c'est le
    test de fumée classique du deep learning (« overfit a single batch »). Il vérifie que le
    graphe d'autograd relie réellement la perte à tous les poids, que l'optimiseur les met à
    jour, et que la perte initiale correspond à ce que la théorie annonce : un modèle
    fraîchement initialisé ne sait rien, sa distribution est quasi uniforme sur les 16 tokens,
    donc son entropie croisée vaut environ `ln(16) ≈ 2.7726`. Cette valeur de référence est
    l'outil de diagnostic le plus rentable du métier : une perte initiale très différente
    signale une mauvaise initialisation, un mauvais alignement des cibles ou un `vocab_size`
    erroné.

    Schéma mental
    -------------
        données : token_ids (batch=1, seq=4) et targets (1, 4) = ids décalés d'une position
        modèle  : vocab_size=16, hidden=8, num_layers=2, num_heads=2   (graine 0)

        pas 0  : cross_entropy(logits.reshape(4, 16), targets.reshape(4)) ~= ln(16) = 2.7726
        ...     50 pas de descente de gradient sur CE SEUL exemple
        pas 49 : perte nettement plus basse (le modèle apprend par cœur)

        losses = [2.77, ..., < 2.77]     losses[-1] < losses[0]

    Ce que ce test vérifie
    ----------------------
    1. la boucle rend l'historique complet des pertes : une valeur par pas, soit 50 valeurs,
       et 1 seule valeur si on ne demande qu'un pas (un historique, pas seulement la dernière
       perte : sinon on ne peut rien diagnostiquer) ;
    2. toutes les pertes sont des flottants finis et positifs (une entropie croisée négative
       ou `nan` trahit un bug de perte ou de gradient) ;
    3. la perte initiale vaut environ `ln(16) ≈ 2.7726`, l'entropie d'une distribution
       uniforme sur 16 tokens : le modèle part bien de l'ignorance ;
    4. la perte finale est STRICTEMENT inférieure à la perte initiale, et passe sous la
       barre de l'uniforme : l'apprentissage a réellement eu lieu.

    API à faire émerger (cible proposée : `src/inference_lab/training/loop.py`, la roadmap
    n'indique que le package `src/inference_lab/training/`, à créer)
    --------------------------------------------------------------------------------------
        def train_steps(
            model: torch.nn.Module,
            token_ids: torch.Tensor,
            targets: torch.Tensor,
            steps: int,
            learning_rate: float,
        ) -> list[float]: ...

    Indice : boucle `for _ in range(steps)` avec `optimizer.zero_grad()`, forward,
    `torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))`,
    `loss.backward()`, `optimizer.step()`, et `losses.append(loss.item())` — `item()` détache
    la valeur, sinon on conserve tout le graphe en mémoire. Pièges : oublier `zero_grad()`
    (les gradients s'accumulent), appeler `model.eval()` alors qu'on entraîne, et surtout
    aplatir logits et cibles de façon incohérente : `cross_entropy` attend `(N, C)` et `(N,)`.
    """

    pytest.skip("Roadmap TDD 3.11 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.training.loop import train_steps

    # Arrange — fixer la graine avec `torch.manual_seed(0)` AVANT toute création de tenseur,
    #           puis construire :
    #           - `model`, le `MinimalTransformer` de la section 3.3 (vocab_size=16,
    #             hidden_size=8, num_layers=2, num_heads=2), en mode entraînement ;
    #           - `token_ids`, un tenseur `torch.long` de shape (batch=1, seq=4) d'ids dans
    #             [0, 16), et `targets`, de même shape, contenant la suite à prédire (les mêmes
    #             ids décalés d'une position) ;
    #           - `steps = 50` et `learning_rate = 0.1`, assez pour que ce modèle minuscule
    #             apprenne par cœur cet unique exemple.

    # Act — lancer la boucle d'entraînement sur ces données et récupérer `losses`,
    #       l'historique des pertes pas à pas.

    # Assert 1 — un historique complet, une perte par pas, quel que soit le nombre de pas
    assert isinstance(losses, list)
    assert len(losses) == 50
    assert len(train_steps(model, token_ids, targets, steps=1, learning_rate=0.1)) == 1

    # Assert 2 — des pertes finies et positives
    assert all(isinstance(loss, float) for loss in losses)
    assert bool(torch.isfinite(torch.tensor(losses)).all())
    assert min(losses) > 0.0

    # Assert 3 — au départ, le modèle est ignorant : entropie de l'uniforme sur 16 tokens
    assert losses[0] == pytest.approx(2.7726, abs=0.5)

    # Assert 4 — la perte descend vraiment, et passe sous la barre de l'uniforme
    assert losses[-1] < losses[0]
    assert losses[-1] < 2.7726
