"""Section 2.1 — projections linéaires Q, K, V des activations d'entrée.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/02_attention/test_qkv_projection.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
((1, 4, 8), 8 = 2 x 4, ...) : c'est volontaire. Un test doit énoncer la vérité attendue,
pas la recalculer avec la même formule que le code testé.

Toute la section 2 travaille avec le même jeu de dimensions minuscule : batch=1, seq=4,
hidden=8, num_heads=2, head_dim=4. Cette première étape installe l'entrée de l'attention :
sans Q, K et V, ni les scores (2.2) ni le KV cache (section 4) n'ont de sens.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_qkv_projections_produce_expected_shapes():
    """Roadmap 2.1 — un même vecteur de token joue trois rôles obtenus par trois matmuls.

    Objectif d'apprentissage
    ------------------------
    Un token entre dans le bloc sous la forme d'UN seul vecteur de taille `hidden`.
    L'attention a besoin de trois rôles distincts pour ce vecteur : ce que le token
    cherche (`Q`), ce par quoi il se laisse trouver (`K`), et ce qu'il transmet (`V`).
    Ces trois rôles viennent de trois matrices de poids indépendantes, donc de trois
    produits matriciels. C'est aussi ici que naît le KV cache : seuls `K` et `V` peuvent
    être mémorisés d'un token sur l'autre, jamais `Q`, recalculé à chaque nouvelle requête.

    Schéma mental
    -------------
        x (batch=1, seq=4, hidden=8)
            --W_q (8, 8)--> Q (1, 4, 8)
            --W_k (8, 8)--> K (1, 4, 8)
            --W_v (8, 8)--> V (1, 4, 8)

        plus loin (2.7) : (1, 4, 8) --view--> (1, 4, num_heads=2, head_dim=4)

    Ce que ce test vérifie
    ----------------------
    1. les trois projections rendent la shape (1, 4, 8) et conservent le dtype d'entrée ;
    2. une projection n'est qu'un matmul : projeter par la matrice identité rend `x` ;
    3. trois matrices différentes donnent trois tenseurs différents (les rôles ne sont
       pas interchangeables) ;
    4. hidden se découpe exactement en num_heads x head_dim = 2 x 4 = 8.

    API à faire émerger (cible roadmap : `src/inference_lab/nn/attention/qkv.py`)
    ---------------------------------------------------------------------------
        def project_qkv(
            x: torch.Tensor, w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    Indice : convention retenue ici, `x @ w` avec `w` de shape (hidden, hidden). Piège à
    connaître : `torch.nn.Linear` stocke sa matrice transposée (out_features, in_features)
    et calcule `x @ w.T`. `torch.eye(8)` sert d'oracle : projeter par l'identité ne doit
    rien changer.
    """

    pytest.skip("Roadmap TDD 2.1 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.attention.qkv import project_qkv

    # Arrange — dimensions de toute la section : batch=1, seq=4, hidden=8, num_heads=2,
    #           head_dim=4.
    #           `x` : activations d'entrée de shape (1, 4, 8) en `torch.float32`, valeurs
    #           quelconques mais déterministes (seed fixée).
    #           `w_q`, `w_k`, `w_v` : trois matrices de poids (8, 8) DIFFÉRENTES entre elles,
    #           aucune nulle, seed fixée.
    #           `w_id` : la matrice identité (8, 8).

    # Act — appeler `project_qkv(x, w_q, w_k, w_v)` et nommer les résultats `q`, `k`, `v` ;
    #       rappeler la même fonction en passant `w_id` pour les trois poids et nommer les
    #       résultats `q_id`, `k_id`, `v_id`.

    # Assert 1 — les trois projections conservent shape et dtype
    assert len(project_qkv(x, w_q, w_k, w_v)) == 3
    assert q.shape == (1, 4, 8)
    assert k.shape == (1, 4, 8)
    assert v.shape == (1, 4, 8)
    assert q.dtype is torch.float32

    # Assert 2 — projeter par l'identité ne change rien : c'est bien un matmul
    torch.testing.assert_close(q_id, x, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(v_id, x, rtol=1e-6, atol=1e-6)

    # Assert 3 — trois matrices distinctes produisent trois tenseurs distincts
    assert not torch.allclose(q, k)
    assert not torch.allclose(k, v)

    # Assert 4 — hidden se découpe exactement en têtes : 8 = 2 x 4
    assert q.shape[-1] == 2 * 4
    assert q.view(1, 4, 2, 4).shape == (1, 4, 2, 4)
