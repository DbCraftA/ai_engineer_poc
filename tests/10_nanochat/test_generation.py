"""Section 10.4 — génération sur l'architecture moderne : exactement max_new_tokens ids valides.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/10_nanochat/test_generation.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(6 + 4 = 10 tokens, (1, 7), ids dans [0, 32)) : c'est volontaire. Un test doit énoncer la
vérité attendue, pas la recalculer avec la même formule que le code testé.

Modèle jouet de la section 10, constant dans tout le chapitre : `vocab_size=32`,
`hidden_size=16`, `num_layers=2`, `num_heads=4`, `num_kv_heads=2`, `head_dim=4`,
`intermediate_size=32`, `seq=6` (longueur du prompt), `batch=1`. Ce fichier ne définit AUCUNE
nouvelle API : il réutilise `generate` de `src/inference_lab/inference/generation.py`, déjà
spécifiée par 3.10 (forme « ids seuls », avec `eos_token_id`) et par 4.6 (forme « ids +
logits par étape », avec `use_cache`). Le point à retenir de la section : une boucle de
génération correctement écrite ne connaît rien de l'architecture qu'elle pilote — passer du
transformer jouet de la section 3 au modèle RMSNorm/RoPE/GQA/SwiGLU ne change pas une ligne
de la boucle.

Convention retenue pour l'arrêt, comme en 3.10 : le token EOS généré est CONSERVÉ dans la
sortie, puis la boucle s'arrête immédiatement. Ici `eos_token_id` est laissé à `None` : on
veut mesurer le budget de tokens, pas l'arrêt anticipé.

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
def test_nanochat_like_model_can_generate_tokens():
    """Roadmap 10.4 — la boucle autorégressive est indépendante de l'architecture du modèle.

    Objectif d'apprentissage
    ------------------------
    Générer, c'est boucler : lire les logits de la dernière position, choisir un id, le
    concaténer, recommencer. La longueur de sortie est donc parfaitement prévisible,
    `len(prompt) + max_new_tokens`, et chaque id produit doit être un indice valide du
    vocabulaire — un id hors de `[0, vocab_size)` ferait planter l'embedding à l'itération
    suivante, ou pire, indexerait un token qui n'existe pas côté tokenizer.

    L'intérêt de rejouer ce test ici est l'INDÉPENDANCE : la boucle écrite en 3.10 pour un
    transformer jouet doit fonctionner telle quelle sur une architecture moderne complète.
    C'est le signe que la frontière est au bon endroit — le modèle rend des logits, le moteur
    fait le reste. C'est aussi ce qui rend la suite possible : échantillonnage (3.6-3.9), KV
    cache (10.5), batching (section 6) et backends (section 17) se branchent sur cette même
    frontière, sans jamais rouvrir le modèle.

    Schéma mental
    -------------
        prompt (batch=1, seq=6)                        max_new_tokens = 4

        itération 1 : (1, 6) -> logits (1, 6, 32) -> logits[:, -1, :] -> +1 id -> (1, 7)
        itération 2 : (1, 7) -> logits (1, 7, 32) -> logits[:, -1, :] -> +1 id -> (1, 8)
        itération 3 : (1, 8) -> logits (1, 8, 32) -> logits[:, -1, :] -> +1 id -> (1, 9)
        itération 4 : (1, 9) -> logits (1, 9, 32) -> logits[:, -1, :] -> +1 id -> (1, 10)

        sortie (1, 10) = 6 + 4     les 6 premières colonnes SONT le prompt
                                   les 4 dernières sont des ids dans [0, 32)

    Ce que ce test vérifie
    ----------------------
    1. la sortie est un tenseur d'ids `torch.long` de shape `(1, 10)`, soit exactement
       `len(prompt) + max_new_tokens = 6 + 4` ;
    2. exactement `max_new_tokens = 4` tokens ont été AJOUTÉS, et le prompt est conservé tel
       quel en préfixe : la génération allonge la séquence, elle ne la réécrit pas ;
    3. tous les tokens générés sont des indices valides du vocabulaire, dans `[0, 32)` ;
    4. la boucle avance d'exactement un token par itération (`max_new_tokens=1` rend `(1, 7)`)
       et le greedy est déterministe : deux appels identiques rendent la même séquence.

    API à faire émerger (cible roadmap « generation », cible proposée :
    `src/inference_lab/inference/generation.py`, module déjà visé par 3.10 et 4.6)
    -----------------------------------------------------------------------------
        def generate(
            model: torch.nn.Module,
            token_ids: torch.Tensor,
            max_new_tokens: int,
            eos_token_id: int | None = None,
            use_cache: bool = False,
            return_step_logits: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

    C'est la réunion des deux contrats existants : 3.10 exige la forme « ids seuls » (c'est
    celle utilisée ici, avec les valeurs par défaut), 4.6 la forme « `(output_ids,
    step_logits)` » — d'où le drapeau `return_step_logits`, que 10.5 activera pour comparer
    les logits avec et sans cache.

    Indice : rien de neuf à écrire si 3.10 est au vert ; vérifie seulement que ta boucle ne
    suppose aucune propriété du modèle au-delà de `logits = model(token_ids)`. Pièges : lire
    `logits[:, -1, :]` et non `logits[:, 0, :]`, conserver `dtype=torch.long` au moment du
    `cat`, ne pas modifier le tenseur d'entrée en place, et envelopper la boucle dans
    `torch.no_grad()` avec le modèle en `eval()`.
    """

    pytest.skip("Roadmap TDD 10.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.inference.generation import generate

    # Arrange — fixer la graine (`torch.manual_seed(0)`) puis construire :
    #           - `model`, le `NanochatLikeModel` de la section 10.2 (vocab_size=32,
    #             hidden_size=16, num_layers=2, num_heads=4, num_kv_heads=2, head_dim=4,
    #             intermediate_size=32), en mode évaluation et sans dropout ;
    #           - `prompt`, un tenseur `torch.long` de shape (batch=1, seq=6) dont les ids
    #             sont dans [0, 32) ;
    #           - `max_new_tokens = 4`, et aucun id d'arrêt (`eos_token_id` laissé à None)
    #             pour que la boucle consomme tout son budget.

    # Act — appeler la génération greedy sur `model` et `prompt` pour obtenir `output`,
    #       le tenseur des ids complets (prompt suivi des tokens générés).

    # Assert 1 — longueur finale = longueur du prompt + tokens demandés
    assert output.shape == (1, 10)
    assert output.dtype is torch.long

    # Assert 2 — exactement max_new_tokens ajoutés, prompt conservé en préfixe
    assert output.shape[1] - prompt.shape[1] == 4
    assert torch.equal(output[:, :6], prompt)

    # Assert 3 — les tokens générés sont des ids valides du vocabulaire
    assert int(output[:, 6:].min()) >= 0
    assert int(output[:, 6:].max()) < 32

    # Assert 4 — un seul token par itération, et un greedy reproductible
    assert generate(model, prompt, max_new_tokens=1).shape == (1, 7)
    assert torch.equal(generate(model, prompt, max_new_tokens=4), output)
