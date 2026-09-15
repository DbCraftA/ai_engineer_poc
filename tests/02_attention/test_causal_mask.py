"""Section 2.4 — masque causal : interdire à un token de regarder le futur.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/02_attention/test_causal_mask.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(10 positions autorisées, 1 seul poids non nul sur la ligne 0, ...) : c'est volontaire. Un
test doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Dimensions constantes de la section : batch=1, seq=4, hidden=8, num_heads=2, head_dim=4.
Le masque s'applique aux scores de 2.3 et conditionne le softmax de 2.5 : c'est lui qui rend
la génération auto-régressive possible, et c'est lui que le KV cache exploitera en decode.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_causal_mask_prevents_future_tokens_from_contributing():
    """Roadmap 2.4 — mettre les positions futures à -inf AVANT le softmax les annule.

    Objectif d'apprentissage
    ------------------------
    Un modèle de langage prédit le token suivant : pendant l'entraînement comme pendant le
    prefill, tous les tokens sont traités en parallèle, mais le token d'indice i ne doit
    voir que les indices 0..i. On n'enlève pas les colonnes interdites, on met leur score à
    -inf : `exp(-inf) = 0`, donc le softmax leur donne une probabilité exactement nulle
    tout en gardant une matrice rectangulaire, donc un matmul dense efficace.
    En decode avec KV cache, la requête est le dernier token : il voit tout le passé et le
    masque devient inutile — comprendre le masque, c'est comprendre pourquoi.

    Schéma mental
    -------------
        seq = 4, masque booléen (True = autorisé), 1 + 2 + 3 + 4 = 10 positions autorisées

            k0 k1 k2 k3
        q0   1  0  0  0
        q1   1  1  0  0
        q2   1  1  1  0
        q3   1  1  1  1

        scores (1, 2, 4, 4) --masque--> triangle strictement supérieur = -inf
        softmax(dim=-1)     --------->  nombre de poids non nuls par ligne : [1, 2, 3, 4]
        ligne q0 : un seul poids non nul, qui vaut donc 1.0

    Ce que ce test vérifie
    ----------------------
    1. le masque est booléen, triangulaire inférieur, et autorise 10 positions sur 16
       (et 3 sur 4 pour seq=2) ;
    2. appliquer le masque met -inf sur les 6 positions futures de chaque tête (12 au total)
       et laisse les scores passés intacts ;
    3. après softmax, la ligne 0 n'a qu'UN seul poids non nul, égal à 1.0 ;
    4. le nombre de poids non nuls par ligne est exactement [1, 2, 3, 4].

    API à faire émerger (cible roadmap : `src/inference_lab/nn/attention/mask.py`)
    ----------------------------------------------------------------------------
        def causal_mask(seq_len: int) -> torch.Tensor: ...            # (seq_len, seq_len) bool
        def apply_causal_mask(scores: torch.Tensor) -> torch.Tensor: ...

    Indice : `torch.ones(4, 4, dtype=torch.bool).tril()` construit le masque,
    `scores.masked_fill(~mask, float("-inf"))` l'applique en diffusant sur (batch, heads).
    Pièges : utiliser `-1e9` au lieu de `-inf` laisse une probabilité minuscule mais non
    nulle ; `masked_fill_` modifierait `scores` en place et fausserait l'assert 2.
    """

    pytest.skip("Roadmap TDD 2.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.attention.mask import apply_causal_mask, causal_mask

    # Arrange — batch=1, num_heads=2, seq=4, head_dim=4.
    #           `scores` : tenseur (1, 2, 4, 4) en `torch.float32`, toutes valeurs FINIES et
    #           déterministes (seed fixée), aucune valeur infinie au départ.

    # Act — construire `mask = causal_mask(4)` puis `masked = apply_causal_mask(scores)`, et
    #       en déduire `probs` avec `torch.softmax(masked, dim=-1)`.

    # Assert 1 — masque booléen triangulaire inférieur : 1 + 2 + 3 + 4 = 10 positions permises
    assert mask.shape == (4, 4)
    assert mask.dtype is torch.bool
    assert int(mask.sum()) == 10
    assert int(causal_mask(2).sum()) == 3
    assert bool(mask[0, 0]) and not bool(mask[0, 1])

    # Assert 2 — le futur est mis à -inf, le passé n'est pas touché
    assert masked.shape == (1, 2, 4, 4)
    assert int(apply_causal_mask(scores).isinf().sum()) == 12
    assert masked[0, 0, 0, 1] == float("-inf")
    assert masked[0, 0, 1, 0] == scores[0, 0, 1, 0]
    assert masked[0, 1, 3, 3] == scores[0, 1, 3, 3]

    # Assert 3 — la ligne 0 ne peut regarder qu'elle-même : un seul poids, qui vaut 1.0
    assert int((probs[0, 0, 0] > 0).sum()) == 1
    torch.testing.assert_close(probs[0, 0, 0, 0], torch.tensor(1.0), atol=1e-6, rtol=0.0)

    # Assert 4 — le nombre de clés visibles croît d'un token par ligne
    assert (probs[0, 0] > 0).sum(dim=-1).tolist() == [1, 2, 3, 4]
