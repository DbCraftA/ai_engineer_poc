"""Section 9.3 — réduction en Triton : la somme est juste, mais pas au bit près.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/09_triton/test_reduction.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(scalaire de shape (), tolérances rtol=1e-5 / atol=1e-5, somme exacte 1000.0) : c'est
volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule
que le code testé.

Cap du chapitre : le kernel répond à une question mémoire, pas à une envie de vitesse, et le
test ne mesure aucune performance — il vérifie la CORRECTION contre PyTorch, notre oracle.
Cette section est la charnière du chapitre : en 9.2, une addition élémentaire donnait un
résultat bit à bit identique à PyTorch. Ici, dès qu'on additionne plusieurs termes, l'ordre de
sommation change (par blocs, en parallèle, en arbre) et l'addition flottante n'est pas
associative : exiger `torch.equal` serait une erreur de méthode. On passe donc à
`torch.testing.assert_close` avec une tolérance, méthode qui restera la règle en 9.4 et 9.5.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
@pytest.mark.triton
def test_triton_reduction_matches_pytorch():
    """Roadmap 9.3 — réduire, c'est changer l'ordre des additions, donc les derniers bits.

    Objectif d'apprentissage
    ------------------------
    La réduction est le motif qui sous-tend tout un moteur d'inférence : produit scalaire d'un
    matmul, somme des exponentielles d'un softmax, moyenne des carrés d'un RMSNorm. Elle
    introduit deux difficultés absentes de 9.2. D'abord la coopération : un program réduit son
    bloc avec `tl.sum`, puis les résultats partiels doivent être combinés entre programs
    (`tl.atomic_add` sur un accumulateur, ou une seconde passe). Ensuite le masquage
    ARITHMÉTIQUE : les cases hors tenseur doivent être chargées avec `other=0.0`, l'élément
    neutre de l'addition ; charger n'importe quoi, ou laisser la valeur par défaut sur un
    accumulateur non initialisé, pollue le total sans provoquer la moindre erreur mémoire.

    Le point à retenir : l'ordre de sommation du GPU diffère de celui de PyTorch, et
    l'addition fp32 n'est pas associative — (a + b) + c n'est pas bit à bit égal à a + (b + c).
    Le résultat est donc juste « à une tolérance près », pas identique. C'est exactement la
    raison pour laquelle tout le dépôt compare avec `assert_close` et une tolérance, et non
    avec `torch.equal`.

    Schéma mental
    -------------
        x (1024,) float32, valeurs > 0 de magnitude ~1  ->  somme ~1000

        PyTorch : ordre de réduction interne          ->  référence x.sum()
        Triton  : 4 blocs de 256 réduits en parallèle, puis combinés
                  ordre différent  ->  écart de l'ordre de 1e-6 sur une somme de ~1000

        assert_close(total, x.sum(), rtol=1e-5, atol=1e-5)  -> OK
        torch.equal(total, x.sum())                         -> à ne PAS exiger

        cas exact : 1000 fois 1.0 -> 1000.0 quel que soit l'ordre (entiers représentables)

    Ce que ce test vérifie
    ----------------------
    1. une réduction rend un SCALAIRE (shape ()), en float32, sur `"cuda"` : la dimension
       réduite disparaît ;
    2. le total correspond à `x.sum()` de PyTorch avec rtol=1e-5 et atol=1e-5 — une tolérance,
       pas une égalité bit à bit, contrairement à l'addition élémentaire de 9.2 ;
    3. avec une taille non multiple du bloc (1000 pour un bloc de 256), le total reste juste :
       les cases masquées ont bien été chargées à 0.0, l'élément neutre ;
    4. sur un cas exactement représentable en binaire — 1000 valeurs égales à 1.0 — le total
       vaut exactement 1000.0, quel que soit l'ordre de sommation.

    API à faire émerger (cible roadmap : `src/inference_lab/kernels/triton/reduction.py`)
    -----------------------------------------------------------------------------------
        def triton_sum(x: torch.Tensor) -> torch.Tensor: ...

    Indice : dans le kernel, `tl.load(x_ptr + offs, mask=mask, other=0.0)` puis
    `tl.sum(vals, axis=0)`, et `tl.atomic_add(out_ptr, partiel)` pour agréger les blocs. Côté
    wrapper, l'accumulateur doit être créé avec `torch.zeros((), device=..., dtype=...)` :
    `torch.empty` laisserait une valeur arbitraire dans le total. Pièges : oublier `other=0.0`
    (le total dépend alors du contenu résiduel de la mémoire), oublier que `x.sum()` renvoie un
    tenseur 0-dim et non un `float`, et vouloir resserrer la tolérance à 1e-7 « pour être
    rigoureux » : cela rend le test faux, pas plus rigoureux.
    """

    pytest.skip("Roadmap TDD 9.3 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.kernels.triton.reduction import triton_sum

    # Arrange — appeler `torch.manual_seed(0)` d'abord, puis créer, tous en `torch.float32`,
    #           contigus et alloués directement sur `"cuda"` : `x` de shape (1024,) aux valeurs
    #           STRICTEMENT positives de magnitude proche de 1, pour que la somme (environ
    #           1000) soit bien conditionnée, sans compensation entre termes de signes opposés ;
    #           `x_tail` de shape (1000,) avec la même propriété, taille NON multiple de 256
    #           (1000 = 3 x 256 + 232) ; `ones` de shape (1000,) entièrement rempli de 1.0.

    # Act — calculer `total = triton_sum(x)`, `total_tail = triton_sum(x_tail)` et
    #       `total_ones = triton_sum(ones)`, le kernel réduisant par blocs de 256 éléments.

    # Assert 1 — réduire fait disparaître la dimension : le résultat est un scalaire
    assert total.dim() == 0
    assert total.shape == ()
    assert total.dtype is torch.float32
    assert total.device.type == "cuda"

    # Assert 2 — même somme que PyTorch, à la tolérance fp32 près : l'ordre diffère
    torch.testing.assert_close(total, x.sum(), rtol=1e-5, atol=1e-5)

    # Assert 3 — taille non multiple du bloc : les cases masquées valent 0.0
    torch.testing.assert_close(total_tail, x_tail.sum(), rtol=1e-5, atol=1e-5)

    # Assert 4 — cas exactement représentable : 1000 x 1.0 = 1000.0, sans arrondi possible
    assert total_ones.item() == 1000.0
    assert triton_sum(ones).item() == 1000.0
