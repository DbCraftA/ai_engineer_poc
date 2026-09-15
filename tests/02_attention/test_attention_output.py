"""Section 2.6 — sortie de l'attention : moyenne pondérée des valeurs V.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/02_attention/test_attention_output.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(probabilité one-hot -> ligne 2 de V, probabilités 0.25 -> moyenne de V) : c'est volontaire.
Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code
testé.

Dimensions constantes de la section : batch=1, seq=4, hidden=8, num_heads=2, head_dim=4.
C'est le deuxième et dernier matmul de l'attention : il ferme la chaîne
scores (2.2/2.3) -> masque (2.4) -> softmax (2.5) -> sortie (2.6).

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_attention_output_is_weighted_sum_of_values():
    """Roadmap 2.6 — la sortie d'une requête est une combinaison convexe des lignes de V.

    Objectif d'apprentissage
    ------------------------
    Les probabilités de 2.5 ne sont pas la sortie de l'attention : elles servent à mélanger
    les vecteurs `V`. Chaque token requête récupère `probs @ V`, c'est-à-dire une moyenne
    pondérée des valeurs des tokens qu'il lit. Comme les poids sont positifs et somment à 1,
    la sortie reste dans l'enveloppe convexe de V : pas d'explosion d'amplitude, ce qui
    explique la stabilité de l'attention même en précision réduite. La dernière dimension de
    la sortie vient de V (`head_dim`), pas des probabilités.

    Schéma mental
    -------------
        probs (batch=1, heads=2, seq_q=4, seq_k=4)
        V     (1, 2, seq_k=4, head_dim=4)
        out = probs @ V  ->  (1, 2, 4, 4) = (batch, heads, seq_q, head_dim)

        cas repère 1 : ligne de probs = [0, 0, 1, 0]  ->  out = V[..., 2, :] exactement
        cas repère 2 : ligne de probs = [0.25] * 4    ->  out = moyenne des 4 lignes de V

    Ce que ce test vérifie
    ----------------------
    1. la shape de sortie est (1, 2, 4, 4), sa dernière dimension venant de V ;
    2. cas dégénéré one-hot : la sortie est exactement la ligne 2 de V, recopiée ;
    3. cas uniforme : la sortie est la moyenne des lignes de V ;
    4. dans tous les cas la sortie reste bornée par le min et le max de V (combinaison
       convexe : aucune amplification).

    API à faire émerger (cible roadmap : `src/inference_lab/nn/attention/naive.py`)
    -----------------------------------------------------------------------------
        def attention_output(probs: torch.Tensor, v: torch.Tensor) -> torch.Tensor: ...

    Indice : `probs @ v` (ou `torch.matmul`) suffit ; il n'y a AUCUNE transposition ici,
    contrairement à `Q @ K^T`. Piège : inverser l'ordre (`v @ probs`) passe la vérification
    de shape quand seq == head_dim, comme ici, mais donne un résultat faux — c'est
    exactement ce que traquent les asserts 2 et 3.
    """

    pytest.skip("Roadmap TDD 2.6 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.attention.naive import attention_output

    # Arrange — batch=1, num_heads=2, seq=4, head_dim=4.
    #           `v` : tenseur (1, 2, 4, 4) en `torch.float32`, valeurs toutes distinctes,
    #           positives et négatives, déterministes (seed fixée).
    #           `probs` : tenseur (1, 2, 4, 4) de probabilités valides (positives, chaque ligne
    #           sommant à 1), par exemple issu d'un `torch.softmax(dim=-1)` sur des scores.
    #           `probs_onehot` : (1, 2, 4, 4) où chaque ligne met toute sa masse (1.0) sur la
    #           clé d'indice 2 et 0.0 ailleurs.
    #           `probs_uniform` : (1, 2, 4, 4) rempli de 0.25.

    # Act — calculer `out`, `out_onehot` et `out_uniform` en appelant `attention_output` avec
    #       respectivement `probs`, `probs_onehot` et `probs_uniform`, toujours avec `v`.

    # Assert 1 — (batch, heads, seq_q, head_dim) : la dernière dimension vient de V
    assert out.shape == (1, 2, 4, 4)
    assert attention_output(probs, v).shape == (1, 2, 4, 4)
    assert out.shape[-1] == v.shape[-1]

    # Assert 2 — poids one-hot sur la clé 2 : la sortie est la ligne 2 de V, à l'identique
    torch.testing.assert_close(out_onehot, v[:, :, 2:3, :].expand(1, 2, 4, 4), atol=1e-6, rtol=0.0)

    # Assert 3 — poids uniformes : la sortie est la moyenne des lignes de V
    torch.testing.assert_close(out_uniform[0, 0, 0], v[0, 0].mean(dim=0), atol=1e-6, rtol=1e-6)

    # Assert 4 — combinaison convexe : rien ne sort de l'intervalle des valeurs de V
    assert bool((out <= v.amax(dim=-2, keepdim=True) + 1e-6).all())
    assert bool((out >= v.amin(dim=-2, keepdim=True) - 1e-6).all())
