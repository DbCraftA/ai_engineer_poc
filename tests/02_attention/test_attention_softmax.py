"""Section 2.5 — softmax de l'attention : des scores aux probabilités.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/02_attention/test_attention_softmax.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(somme = 1, uniforme = 0.25 = 1/4, positions masquées = 0.0) : c'est volontaire. Un test doit
énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Dimensions constantes de la section : batch=1, seq=4, hidden=8, num_heads=2, head_dim=4.
Le softmax transforme les scores de 2.3, éventuellement masqués en 2.4, en poids de mélange
utilisés en 2.6. Sa forme stabilisée (soustraction du max) est aussi le point de départ du
softmax « en ligne » de FlashAttention.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_attention_probabilities_sum_to_one():
    """Roadmap 2.5 — chaque requête répartit une masse totale de 1 sur ses clés.

    Objectif d'apprentissage
    ------------------------
    Le softmax est appliqué sur la DERNIÈRE dimension, celle des clés : chaque token requête
    obtient une distribution de probabilité sur les tokens qu'il peut lire. C'est ce qui rend
    la sortie de l'attention (2.6) une moyenne pondérée, donc bornée et stable. Se tromper
    d'axe (`dim=-2` au lieu de `dim=-1`) produit un tenseur de bonne shape mais un modèle
    silencieusement faux : cet axe est l'erreur la plus fréquente de toute la section.

    Schéma mental
    -------------
        scores (batch=1, heads=2, seq=4, seq=4)  --softmax(dim=-1)-->  probs (1, 2, 4, 4)

        probs.sum(dim=-1) == torch.ones(1, 2, 4)     (une masse de 1 par ligne)
        scores tous égaux sur une ligne  ->  1/4 = 0.25 sur chacune des 4 clés
        softmax(scores + 100.0) == softmax(scores)   (invariance par translation)

    Ce que ce test vérifie
    ----------------------
    1. la shape est conservée : (1, 2, 4, 4) ;
    2. la somme sur la dernière dimension vaut exactement 1 pour chaque requête ;
    3. des scores uniformes donnent 0.25 partout, et toutes les probabilités sont dans ]0, 1] ;
    4. ajouter une constante à tous les scores d'une ligne ne change pas les probabilités.

    API à faire émerger (cible roadmap : `src/inference_lab/nn/attention/naive.py`)
    -----------------------------------------------------------------------------
        def attention_probabilities(scores: torch.Tensor) -> torch.Tensor: ...

    Indice : `torch.softmax(scores, dim=-1)` suffit pour le vert, mais écris plutôt la forme
    stabilisée `exp(s - s.max(dim=-1, keepdim=True).values)` normalisée : c'est elle qui
    justifie l'assert 4 et qui reviendra en FP16, où `exp` déborde très vite.
    """

    pytest.skip("Roadmap TDD 2.5 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.attention.naive import attention_probabilities

    # Arrange — batch=1, num_heads=2, seq=4, head_dim=4.
    #           `scores` : tenseur (1, 2, 4, 4) en `torch.float32`, valeurs finies, distinctes
    #           sur chaque ligne, déterministes (seed fixée).
    #           `scores_uniform` : tenseur (1, 2, 4, 4) dont toutes les valeurs sont ÉGALES.
    #           `scores_shifted` : `scores` auquel on ajoute la même grande constante partout
    #           (par exemple 100.0).

    # Act — calculer `probs`, `probs_uniform` et `probs_shifted` en appelant
    #       `attention_probabilities` sur chacun des trois tenseurs de scores.

    # Assert 1 — le softmax ne change pas la shape
    assert probs.shape == (1, 2, 4, 4)
    assert attention_probabilities(scores).shape == scores.shape

    # Assert 2 — une masse totale de 1 par requête, sur l'axe des clés
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(1, 2, 4), atol=1e-6, rtol=0.0)

    # Assert 3 — scores égaux : 1/4 = 0.25 par clé, et jamais de probabilité hors ]0, 1]
    torch.testing.assert_close(probs_uniform, torch.full((1, 2, 4, 4), 0.25), atol=1e-6, rtol=0.0)
    assert bool((probs > 0.0).all())
    assert bool((probs <= 1.0).all())

    # Assert 4 — invariance par translation : la base de la version numériquement stable
    torch.testing.assert_close(probs_shifted, probs, atol=1e-6, rtol=1e-5)


@pytest.mark.tdd
def test_masked_positions_receive_zero_probability():
    """Roadmap 2.5 — une position à -inf reçoit une probabilité exactement nulle.

    Objectif d'apprentissage
    ------------------------
    Le masque de 2.4 n'a de valeur que si le softmax le transforme en zéro EXACT : sinon le
    modèle laisserait fuir de l'information du futur, et une génération auto-régressive ne
    serait plus reproductible token par token. `exp(-inf) = 0` donne ce zéro exact, et la
    masse retirée est automatiquement redistribuée sur les positions autorisées, qui somment
    toujours à 1. Piège à connaître : une ligne ENTIÈREMENT masquée donne 0/0 = NaN, panne
    classique quand on masque aussi le padding.

    Schéma mental
    -------------
        masked_scores (1, 2, 4, 4), triangle strictement supérieur = -inf

            k0   k1   k2   k3                      k0   k1   k2   k3
        q0   s  -inf -inf -inf   --softmax-->  q0  1.0  0.0  0.0  0.0
        q1   s   s   -inf -inf                 q1   p   1-p  0.0  0.0
                                               chaque ligne somme encore à 1.0

    Ce que ce test vérifie
    ----------------------
    1. les positions masquées ont une probabilité exactement 0.0 (pas 1e-9) ;
    2. les lignes somment toujours à 1 : la masse est redistribuée sur le passé ;
    3. aucune valeur NaN ni infinie ne subsiste dans les probabilités ;
    4. la première ligne est exactement le vecteur one-hot [1, 0, 0, 0].

    API à faire émerger (cible roadmap : `src/inference_lab/nn/attention/naive.py`)
    -----------------------------------------------------------------------------
        def attention_probabilities(scores: torch.Tensor) -> torch.Tensor: ...

    Indice : construis `masked_scores` avec `torch.triu(..., diagonal=1)` et
    `masked_fill(..., float("-inf"))`, sans passer par `src/`, pour que ce test ne dépende que
    du softmax. Dans une version stabilisée, veille à ce que `-inf - max` reste `-inf` et non
    `nan` (cas d'une ligne entièrement masquée).
    """

    pytest.skip("Roadmap TDD 2.5 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.attention.naive import attention_probabilities

    # Arrange — batch=1, num_heads=2, seq=4, head_dim=4.
    #           `masked_scores` : tenseur (1, 2, 4, 4) en `torch.float32` dont le triangle
    #           STRICTEMENT supérieur de la dernière matrice (4, 4) vaut `float("-inf")` et
    #           dont les autres valeurs sont finies, distinctes et déterministes (seed fixée).

    # Act — calculer `probs = attention_probabilities(masked_scores)`.

    # Assert 1 — les positions futures sont à zéro exact, pas seulement proches de zéro
    assert probs[0, 0, 0, 1] == 0.0
    assert attention_probabilities(masked_scores)[0, 0, 0, 3] == 0.0
    assert bool((probs[0, 0].triu(diagonal=1) == 0.0).all())

    # Assert 2 — la masse retirée est redistribuée : les lignes somment toujours à 1
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(1, 2, 4), atol=1e-6, rtol=0.0)

    # Assert 3 — pas de NaN ni d'infini introduits par le -inf
    assert not bool(probs.isnan().any())
    assert bool(probs.isfinite().all())

    # Assert 4 — la première requête ne voit qu'elle-même : one-hot exact
    torch.testing.assert_close(
        probs[0, 0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6, rtol=0.0
    )
