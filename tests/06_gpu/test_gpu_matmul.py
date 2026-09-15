"""Section 6.6 — GEMM / GEMV sur GPU : même résultat que le CPU, à une tolérance près.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/06_gpu/test_gpu_matmul.py -k <nom_du_test>`.
              Le test DOIT échouer : les variables de la partie `Arrange` n'existent pas
              encore (`NameError`) — ici il n'y a aucun module `src/` à créer.
2. GREEN    : écrire le minimum de code dans le test, juste assez pour faire passer les
              assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(shapes (64, 32) et (64,), tolérances rtol=1e-4 / atol=1e-5) : c'est volontaire. Un test
doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Le matmul est l'opération qui consomme presque tout le temps d'un LLM : GEMM pendant le
prefill (matrice x matrice), GEMV pendant le decode (matrice x vecteur, mémoire-bound).
Cette section pose la méthode de validation utilisée partout ensuite : le CPU en float32
sert de référence, le GPU doit lui correspondre à une tolérance près — pas bit à bit,
parce que les ordres de réduction diffèrent.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
@pytest.mark.gpu
@pytest.mark.cuda
def test_gpu_matmul_matches_cpu_reference():
    """Roadmap 6.6 — le GPU calcule le même produit matriciel, à l'arrondi près.

    Objectif d'apprentissage
    ------------------------
    Un GPU découpe le produit matriciel en tuiles réparties sur des milliers de threads :
    la somme sur l'axe K n'est donc pas faite dans le même ordre que sur CPU, et
    l'addition flottante n'est pas associative. Le résultat diffère dans les derniers bits.
    C'est pour cela que tout le dépôt compare avec `torch.testing.assert_close` et non avec
    `torch.equal` : exiger l'égalité bit à bit entre deux backends est une erreur de
    méthode, qui rend les tests d'attention et de kernels Triton impossibles à écrire.

    Attention supplémentaire sur les GPU Ampere et plus récents : PyTorch peut exécuter un
    matmul « float32 » en TF32 (10 bits de mantisse) via
    `torch.backends.cuda.matmul.allow_tf32`, ce qui dégrade l'écart d'un ordre de grandeur.
    D'où une tolérance relative de 1e-4 plutôt que le 1e-6 d'un calcul purement FP32.

    Schéma mental
    -------------
        a (64, 128) float32   @   b (128, 32) float32   ->  GEMM (64, 32)
        a (64, 128) float32   @   v (128,)    float32   ->  GEMV (64,)

        CPU : somme des 128 produits dans l'ordre 0..127          -> référence
        GPU : somme par tuiles, en parallèle, ordre différent     -> écart ~1e-5 absolu

        assert_close(gpu.cpu(), cpu, rtol=1e-4, atol=1e-5)  -> OK
        torch.equal(gpu.cpu(), cpu)                         -> à ne PAS exiger

    Ce que ce test vérifie
    ----------------------
    1. le résultat GPU a la shape (64, 32) attendue, reste en float32 et vit sur `"cuda"` ;
    2. les deux côtés partent exactement des mêmes valeurs d'entrée (le transfert est
       exact, cf. 6.5) : la comparaison porte donc bien sur le calcul ;
    3. le GEMM GPU correspond à la référence CPU avec rtol=1e-4 et atol=1e-5 ;
    4. le GEMV GPU rend la shape (64,) et correspond à la même référence CPU, avec la
       même tolérance.

    Cible roadmap : « benchmark only » — aucun module `src/` à écrire
    ---------------------------------------------------------------
    Ce test exerce directement `torch` : il doit passer au vert dès que le `pytest.skip`
    est retiré et que les variables d'`Arrange` existent.

    Indice : `torch.manual_seed(0)` puis `torch.randn(...)` sur CPU, `.to("cuda")` pour les
    copies GPU, et `@` (ou `torch.matmul`) pour les deux produits. Piège : ne recrée pas les
    tenseurs GPU avec un second `torch.randn(..., device="cuda")` — le générateur CUDA est
    distinct de celui du CPU, tu comparerais deux jeux de valeurs différents.
    """

    pytest.skip("Roadmap TDD 6.6 — supprimer cette ligne pour démarrer le cycle RED")

    # Arrange — avec une seed fixée, créer sur CPU en `torch.float32` : `a_cpu` de shape
    #           (64, 128), `b_cpu` de shape (128, 32) et `v_cpu` de shape (128,). En faire
    #           les copies GPU `a_gpu`, `b_gpu` et `v_gpu` par transfert (pas par un nouveau
    #           tirage aléatoire).

    # Act — calculer la référence CPU `gemm_cpu` = `a_cpu` @ `b_cpu` et `gemv_cpu` =
    #       `a_cpu` @ `v_cpu`, puis les mêmes produits sur GPU dans `gemm_gpu` et `gemv_gpu`.

    # Assert 1 — shape, dtype et device du résultat GPU
    assert gemm_gpu.shape == (64, 32)
    assert gemm_gpu.dtype is torch.float32
    assert gemm_gpu.device.type == "cuda"

    # Assert 2 — les deux côtés calculent sur les mêmes entrées, bit à bit
    assert torch.equal(a_gpu.cpu(), a_cpu)
    assert torch.equal(b_gpu.cpu(), b_cpu)

    # Assert 3 — le GEMM GPU correspond à la référence CPU, à la tolérance près
    torch.testing.assert_close(gemm_gpu.cpu(), gemm_cpu, rtol=1e-4, atol=1e-5)

    # Assert 4 — le GEMV (le cas du decode) suit la même règle
    assert gemv_gpu.shape == (64,)
    torch.testing.assert_close(gemv_gpu.cpu(), gemv_cpu, rtol=1e-4, atol=1e-5)
