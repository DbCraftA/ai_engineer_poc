"""Section 2.11 — RMSNorm : normaliser par la moyenne quadratique, sans centrage.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/02_attention/test_rmsnorm.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(RMS de sortie = 1, poids x2 -> sortie x2, entrée nulle -> sortie nulle) : c'est volontaire. Un
test doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé,
sauf ici pour la formule de référence, écrite explicitement à la main dans l'assert.

Dimensions constantes de la section : batch=1, seq=4, hidden=8, num_heads=2, head_dim=4.
RMSNorm est la normalisation utilisée par Llama et Qwen à la place de LayerNorm : moins
d'opérations, pas de moyenne à soustraire, pas de biais. Elle encadre l'attention (2.13) et
sera le premier module dont les poids seront chargés depuis Hugging Face.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_rmsnorm_preserves_shape():
    """Roadmap 2.11 — normalisation par token : la shape et le dtype ne changent pas.

    Objectif d'apprentissage
    ------------------------
    RMSNorm agit token par token, sur la dernière dimension seulement : chaque vecteur de
    taille `hidden` est ramené à une moyenne quadratique de 1, puis remis à l'échelle par un
    poids appris, un par canal. Aucun mélange entre tokens (contrairement à l'attention),
    aucune statistique de batch (contrairement à BatchNorm) : c'est ce qui la rend compatible
    avec le decode token par token et avec un KV cache. Conséquence pratique : elle est
    négligeable en FLOPs mais visible en bande passante, car elle relit toutes les activations.

    Schéma mental
    -------------
        x (batch=1, seq=4, hidden=8), weight (8,)

        pour chaque token : rms = sqrt(mean(x^2) + eps)  ->  x / rms  ->  x * weight
        sortie (1, 4, 8), identique en shape et dtype

        weight = ones  ->  sqrt(mean(sortie^2)) == 1 pour chacun des 4 tokens
        x tout positif ->  sortie toute positive (aucun centrage, à la différence de LayerNorm)
        x -> 2x        ->  sortie inchangée (invariance d'échelle)

    Ce que ce test vérifie
    ----------------------
    1. shape (1, 4, 8) et dtype float32 conservés ;
    2. avec un poids unitaire, la moyenne quadratique de chaque token de sortie vaut 1 ;
    3. une entrée strictement positive donne une sortie strictement positive : il n'y a PAS de
       soustraction de moyenne ;
    4. multiplier l'entrée par 2 ne change pas la sortie (invariance d'échelle).

    API à faire émerger (cible roadmap : `src/inference_lab/nn/normalization/rmsnorm.py`)
    -----------------------------------------------------------------------------------
        def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor: ...

    Indice : `x.pow(2).mean(dim=-1, keepdim=True)` puis `torch.rsqrt(... + eps)`. Pièges :
    normaliser sur `dim=0` (le batch) ou sur toutes les dimensions à la fois, oublier
    `keepdim=True` (la diffusion devient silencieusement fausse), et soustraire la moyenne
    comme le ferait LayerNorm.
    """

    pytest.skip("Roadmap TDD 2.11 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.normalization.rmsnorm import rms_norm

    # Arrange — batch=1, seq=4, hidden=8.
    #           `x` : tenseur (1, 4, 8) en `torch.float32`, déterministe (seed fixée), valeurs
    #           de magnitude proche de 1, positives ET négatives, aucun token entièrement nul.
    #           `x_positive` : tenseur (1, 4, 8) aux valeurs toutes STRICTEMENT positives.
    #           `w_ones` : tenseur de poids (8,) rempli de 1.

    # Act — calculer `out = rms_norm(x, w_ones)`, `out_positive = rms_norm(x_positive, w_ones)`
    #       et `out_scaled` en appelant `rms_norm` sur `x` multiplié par 2, toujours `w_ones`.

    # Assert 1 — normalisation « sur place » au sens des shapes
    assert out.shape == (1, 4, 8)
    assert rms_norm(x, w_ones).shape == x.shape
    assert out.dtype is torch.float32

    # Assert 2 — moyenne quadratique ramenée à 1 pour chacun des 4 tokens
    torch.testing.assert_close(
        out.pow(2).mean(dim=-1).sqrt(), torch.ones(1, 4), atol=1e-4, rtol=1e-4
    )

    # Assert 3 — aucun centrage : le signe des valeurs est préservé
    assert bool((out_positive > 0.0).all())
    assert bool((out_positive.mean(dim=-1) > 0.0).all())

    # Assert 4 — invariance d'échelle : diviser par la RMS annule le facteur 2
    torch.testing.assert_close(out_scaled, out, atol=1e-5, rtol=1e-4)


@pytest.mark.tdd
def test_rmsnorm_matches_reference_formula():
    """Roadmap 2.11 — x / sqrt(mean(x^2) + eps) * weight, et rien d'autre.

    Objectif d'apprentissage
    ------------------------
    Connaître la formule EXACTE compte : c'est elle qui décidera si notre modèle reproduit la
    référence Hugging Face au 1e-5 près ou s'il en dérive. Deux différences avec LayerNorm sont
    à retenir : pas de soustraction de la moyenne, et pas de biais additif. `eps` n'est pas un
    détail décoratif : il évite la division par zéro sur un token nul, cas qui arrive avec du
    padding ou un embedding non initialisé.

    Schéma mental
    -------------
        RMSNorm  : y = x / sqrt(mean(x^2) + eps) * weight
        LayerNorm: y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias

        x (1, 4, 8), weight (8,), eps = 1e-6

        weight = 2 * ones  ->  sortie = 2 x sortie obtenue avec ones
        x = zeros          ->  sortie = zeros (grâce à eps), aucun NaN

    Ce que ce test vérifie
    ----------------------
    1. la sortie est égale à la formule de référence recalculée à la main dans l'assert ;
    2. le poids agit canal par canal, multiplicativement : doubler le poids double la sortie ;
    3. le résultat DIFFÈRE de `torch.nn.functional.layer_norm` sur une entrée de moyenne non
       nulle (pas de centrage) ;
    4. sur une entrée nulle, `eps` empêche la division par zéro : sortie nulle, aucun NaN.

    API à faire émerger (cible roadmap : `src/inference_lab/nn/normalization/rmsnorm.py`)
    -----------------------------------------------------------------------------------
        def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor: ...

    Indice : place `eps` SOUS la racine (`sqrt(mean + eps)`), pas à côté (`sqrt(mean) + eps`) :
    les deux se ressemblent mais divergent sur les petites valeurs, et Hugging Face utilise la
    première forme. Attention aussi à l'ordre : la division précède la multiplication par le
    poids.
    """

    pytest.skip("Roadmap TDD 2.11 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.normalization.rmsnorm import rms_norm

    # Arrange — batch=1, seq=4, hidden=8.
    #           `x` : tenseur (1, 4, 8) en `torch.float32`, déterministe (seed fixée), dont la
    #           moyenne par token est NON nulle (sinon l'assert 3 ne distingue rien).
    #           `weight` : tenseur (8,) aux valeurs distinctes et non nulles.
    #           `w_ones` : tenseur (8,) rempli de 1.
    #           `x_zeros` : tenseur (1, 4, 8) entièrement nul.
    #           `eps` : le flottant 1e-6, passé explicitement à chaque appel.

    # Act — calculer `out = rms_norm(x, weight, eps)`, `out_ones = rms_norm(x, w_ones, eps)`,
    #       `out_double` en doublant `w_ones`, et `out_zeros = rms_norm(x_zeros, w_ones, eps)`.

    # Assert 1 — la formule de référence, écrite explicitement
    torch.testing.assert_close(
        out,
        x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight,
        atol=1e-6,
        rtol=1e-6,
    )

    # Assert 2 — le poids est un simple facteur par canal
    torch.testing.assert_close(out_double, 2.0 * out_ones, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(rms_norm(x, 2.0 * w_ones, eps), 2.0 * out_ones, atol=1e-6, rtol=1e-6)

    # Assert 3 — ce n'est pas LayerNorm : aucune moyenne n'est soustraite
    assert not torch.allclose(
        out_ones, torch.nn.functional.layer_norm(x, (8,)), atol=1e-4, rtol=1e-4
    )

    # Assert 4 — eps protège le cas dégénéré : pas de 0/0
    torch.testing.assert_close(out_zeros, torch.zeros(1, 4, 8), atol=1e-6, rtol=0.0)
    assert not bool(out_zeros.isnan().any())
