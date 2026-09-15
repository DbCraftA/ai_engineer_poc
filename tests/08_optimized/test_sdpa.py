"""Section 8.1 / 8.2 — attention naïve contre `scaled_dot_product_attention` de PyTorch.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/08_optimized/test_sdpa.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
((1, 2, 4, 4), facteur 0.5 = 1/sqrt(4), première ligne = V[0]) : c'est volontaire. Un test
doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Dimensions constantes, identiques à la section 2 : batch=1, seq=4, hidden=8, num_heads=2,
head_dim=4. Cette section ne fait émerger aucune mathématique nouvelle : elle prouve que le
chemin optimisé de PyTorch calcule EXACTEMENT la même chose que les briques naïves de
`nn/attention/naive.py` et `nn/attention/mask.py`, en une seule passe kernel et sans
matérialiser la matrice (seq, seq). C'est le premier vrai gain d'inférence de la roadmap.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_naive_attention_matches_pytorch_sdpa_for_supported_case():
    """Roadmap 8.1 — SDPA est la même mathématique que l'attention naïve, mieux exécutée.

    Objectif d'apprentissage
    ------------------------
    L'attention naïve de la section 2 s'écrit en quatre appels : `Q @ K^T`, mise à l'échelle,
    `softmax`, `probs @ V`. Chaque appel alloue un tenseur intermédiaire de taille
    (batch, heads, seq, seq) et le relit depuis la mémoire globale : c'est le trafic mémoire,
    pas le calcul, qui domine. `torch.nn.functional.scaled_dot_product_attention` (SDPA) fait
    la même chose en UN seul kernel, sans jamais matérialiser la matrice de scores complète
    (backends « flash » et « memory-efficient »), ce qui rend le coût mémoire linéaire en
    `seq` au lieu de quadratique. Sur Qwen2.5-0.5B en prefill, c'est ce qui distingue un
    prompt de 4 tokens d'un prompt de 4096 tokens. Le test ancre le point crucial : substituer
    SDPA à l'attention naïve est une optimisation, pas un changement de modèle.

    Schéma mental
    -------------
        Q, K, V (batch=1, heads=2, seq=4, head_dim=4), float32

        naïf  : scores (1,2,4,4) -> x 1/sqrt(4)=0.5 -> softmax -> @ V   (4 tenseurs alloués)
        SDPA  : scaled_dot_product_attention(q, k, v)                   (1 kernel, 1 sortie)

        sortie -> (1, 2, 4, 4) = (batch, heads, seq_q, head_dim), identique des deux côtés

        Égalité NUMÉRIQUE, pas bit à bit : SDPA réordonne les réductions et peut fusionner le
        facteur 0.5 dans le matmul, donc l'écart fp32 est ~1e-7. D'où rtol=1e-5 / atol=1e-6.

    Ce que ce test vérifie
    ----------------------
    1. les deux chemins renvoient la même shape (1, 2, 4, 4), le même dtype, et des valeurs
       finies ;
    2. les valeurs coïncident à la tolérance fp32 réaliste rtol=1e-5 / atol=1e-6 ;
    3. le facteur d'échelle implicite de SDPA est bien 1/sqrt(head_dim) = 0.5 : le passer
       explicitement via `scale=` ne change rien ;
    4. cas repère indépendant des deux implémentations : si toutes les lignes de V sont
       identiques, la sortie est cette ligne, quels que soient Q et K (les poids d'attention
       somment à 1).

    API à faire émerger (cible roadmap : « attention », cible proposée
    `src/inference_lab/nn/attention/naive.py`)
    -----------------------------------------------------------------
        def naive_attention(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False
        ) -> torch.Tensor: ...

    Indice : n'écris pas une nouvelle mathématique, compose les briques de la section 2 :
    `attention_output(attention_probabilities(scaled_attention_scores(q, k)), v)`. L'oracle
    est `torch.nn.functional.scaled_dot_product_attention(q, k, v)`, qui attend des tenseurs
    en (batch, heads, seq, head_dim) — c'est déjà la convention de la section 2. Piège : ne
    compare pas avec `torch.equal` ni `atol=0`, SDPA n'est pas bit à bit identique.
    """

    pytest.skip("Roadmap TDD 8.1 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.attention.naive import naive_attention

    # Arrange — batch=1, num_heads=2, seq=4, head_dim=4 (donc facteur attendu 1/sqrt(4) = 0.5).
    #           `q`, `k`, `v` : trois tenseurs (1, 2, 4, 4) en `torch.float32`, contigus,
    #           distincts les uns des autres, non constants, déterministes (seed fixée) et de
    #           magnitude proche de 1 pour que le softmax ne sature pas.
    #           `v_const` : un tenseur (1, 2, 4, 4) dont les 4 lignes de séquence sont
    #           IDENTIQUES au sein de chaque tête (recopie d'une seule ligne de `v`).

    # Act — calculer `out_naive = naive_attention(q, k, v)` et l'oracle
    #       `out_sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v)`, puis les
    #       deux mêmes appels sur `v_const` : `out_naive_const` et `out_sdpa_const`.

    # Assert 1 — même contrat de sortie des deux côtés : (batch, heads, seq_q, head_dim)
    assert out_naive.shape == (1, 2, 4, 4)
    assert out_sdpa.shape == (1, 2, 4, 4)
    assert out_naive.dtype is torch.float32
    assert out_sdpa.dtype is out_naive.dtype
    assert bool(out_naive.isfinite().all())

    # Assert 2 — même mathématique : égalité à la tolérance fp32, pas bit à bit
    torch.testing.assert_close(out_naive, out_sdpa, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(naive_attention(q, k, v), out_sdpa, rtol=1e-5, atol=1e-6)

    # Assert 3 — l'échelle implicite de SDPA est 1/sqrt(head_dim) = 0.5
    torch.testing.assert_close(
        torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=0.5),
        out_sdpa,
        rtol=1e-5,
        atol=1e-6,
    )

    # Assert 4 — V à lignes identiques : la sortie est cette ligne, poids d'attention ou non
    torch.testing.assert_close(out_sdpa_const, v_const, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out_naive_const, v_const, rtol=1e-5, atol=1e-6)


@pytest.mark.tdd
def test_causal_sdpa_matches_naive_causal_attention():
    """Roadmap 8.2 — `is_causal=True` remplace la construction explicite du masque.

    Objectif d'apprentissage
    ------------------------
    En prefill, tous les tokens du prompt passent d'un coup et le masque causal est
    indispensable (section 2.4). Le construire à la main coûte un tenseur (seq, seq) de plus,
    alloué, écrit puis relu — exactement ce que l'on cherche à éviter. `is_causal=True` ne
    construit aucun masque : le kernel saute simplement les blocs situés au-dessus de la
    diagonale, donc il calcule moins ET alloue moins. À seq=4096, le masque explicite pèse à
    lui seul 64 Mio en fp32 par tête. Comprendre que c'est la MÊME opération est ce qui permet
    de remplacer le chemin naïf par le chemin rapide sans jamais changer les sorties du modèle.

    Schéma mental
    -------------
        seq = 4, masque booléen (True = autorisé), 1 + 2 + 3 + 4 = 10 positions permises

            k0 k1 k2 k3
        q0   1  0  0  0     <- q0 ne voit que lui-même : sortie = V[..., 0, :] exactement
        q1   1  1  0  0
        q2   1  1  1  0
        q3   1  1  1  1

        naïf : scores -> masked_fill(~mask, -inf) -> softmax -> @ V
        SDPA : scaled_dot_product_attention(q, k, v, is_causal=True)   (aucun masque alloué)

        les deux -> (1, 2, 4, 4), égaux à ~1e-7 près, et DIFFÉRENTS du cas non causal.

    Ce que ce test vérifie
    ----------------------
    1. l'attention naïve masquée et le SDPA causal coïncident (rtol=1e-5 / atol=1e-6) et
       ne produisent aucun NaN, malgré la ligne 0 remplie de -inf sauf une case ;
    2. cas repère exact : la sortie de la requête 0 est la ligne 0 de V, puisqu'elle n'a
       qu'un seul poids non nul, égal à 1.0 ;
    3. `is_causal=True` équivaut à passer le masque booléen triangulaire inférieur via
       `attn_mask=` : c'est bien le même masque, simplement jamais matérialisé ;
    4. le masque change réellement le résultat : le SDPA causal diffère du SDPA non causal.

    API à faire émerger (cible roadmap : « attention », cible proposée
    `src/inference_lab/nn/attention/naive.py` + `src/inference_lab/nn/attention/mask.py`)
    ------------------------------------------------------------------------------------
        def naive_attention(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False
        ) -> torch.Tensor: ...

    Indice : réutilise `apply_causal_mask` de la section 2.4 entre la mise à l'échelle et le
    softmax ; le masque de PyTorch se reconstruit avec `torch.ones(4, 4, dtype=torch.bool)
    .tril()`. Pièges : un masque `float` s'ADDITIONNE aux scores alors qu'un masque `bool`
    sélectionne les positions autorisées ; et `is_causal=True` aligne la diagonale en haut à
    gauche, donc il est FAUX en decode avec KV cache (1 requête, N clés : la requête ne verrait
    que la clé 0 au lieu de tout le passé). En decode, on n'utilise aucun masque.
    """

    pytest.skip("Roadmap TDD 8.2 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.attention.naive import naive_attention

    # Arrange — batch=1, num_heads=2, seq=4, head_dim=4.
    #           `q`, `k`, `v` : trois tenseurs (1, 2, 4, 4) en `torch.float32`, contigus,
    #           distincts, non constants, déterministes (seed fixée), toutes valeurs finies.
    #           `mask` : le masque booléen (4, 4) triangulaire inférieur (True = position
    #           autorisée), la même vérité que `causal_mask(4)` de la section 2.4.

    # Act — calculer `out_naive_causal = naive_attention(q, k, v, is_causal=True)`, l'oracle
    #       `out_sdpa_causal` avec `scaled_dot_product_attention(..., is_causal=True)`,
    #       `out_sdpa_mask` avec `attn_mask=mask` à la place de `is_causal`, et
    #       `out_sdpa_full` sans aucun masque.

    # Assert 1 — même opération : égalité à la tolérance fp32, et aucun NaN
    assert out_naive_causal.shape == (1, 2, 4, 4)
    torch.testing.assert_close(out_naive_causal, out_sdpa_causal, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        naive_attention(q, k, v, is_causal=True), out_sdpa_causal, rtol=1e-5, atol=1e-6
    )
    assert bool(out_naive_causal.isfinite().all())
    assert bool(out_sdpa_causal.isfinite().all())

    # Assert 2 — la requête 0 ne voit qu'elle-même : sa sortie est la ligne 0 de V
    torch.testing.assert_close(out_sdpa_causal[:, :, 0, :], v[:, :, 0, :], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out_naive_causal[:, :, 0, :], v[:, :, 0, :], rtol=1e-5, atol=1e-6)

    # Assert 3 — `is_causal=True` est exactement le masque triangulaire inférieur
    assert int(mask.sum()) == 10
    torch.testing.assert_close(out_sdpa_mask, out_sdpa_causal, rtol=1e-5, atol=1e-6)

    # Assert 4 — sans masque, les positions futures contribuent : le résultat change
    assert not torch.allclose(out_sdpa_causal, out_sdpa_full, rtol=1e-3, atol=1e-3)
