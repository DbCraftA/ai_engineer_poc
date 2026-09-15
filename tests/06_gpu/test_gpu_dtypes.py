"""Section 6.7 — dtypes sur GPU : FP32, FP16 et BF16 dans un même matmul.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/06_gpu/test_gpu_dtypes.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(shape (64, 32), 2 octets par élément en demi-précision, erreur relative < 1e-3 en FP16 et
< 1e-2 en BF16) : c'est volontaire. Un test doit énoncer la vérité attendue, pas la
recalculer avec la même formule que le code testé.

C'est la suite directe de la section 1.8 (dtype et précision), transposée sur GPU : la
demi-précision divise par deux la mémoire ET le trafic mémoire, ce qui double le débit
théorique d'un decode mémoire-bound. La contrepartie est une erreur d'arrondi mesurable,
que ce test borne au lieu de l'ignorer.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices. Même raison pour l'import
# du module cible : c'est ton code d'`Act` qui l'appellera, le linter le voit donc inutilisé.
# ruff: noqa: F401, F821

import pytest
import torch


@pytest.mark.tdd
@pytest.mark.gpu
@pytest.mark.cuda
def test_fp16_bf16_and_fp32_matmul_preserve_expected_shapes():
    """Roadmap 6.7 — changer de dtype change la précision et la mémoire, jamais la shape.

    Objectif d'apprentissage
    ------------------------
    Quand on charge Qwen2.5-0.5B en FP16 ou BF16, aucune shape ne bouge : ce sont les mêmes
    matrices, stockées sur deux fois moins d'octets. Ce qui change, c'est la précision de
    chaque produit et le dtype de SORTIE, qui suit le dtype d'ENTRÉE — une couche en FP16
    rend des activations FP16, et c'est ainsi que la demi-précision se propage dans tout le
    modèle. Savoir ce que coûte cette propagation en erreur relative est indispensable
    avant de comparer notre implémentation à la référence Hugging Face.

    Rappel de la section 1.8 : FP16 a 11 bits de mantisse et un exposant étroit, BF16 a
    8 bits de mantisse mais l'exposant de FP32 — BF16 est donc moins précis et plus robuste
    aux débordements. Les deux tiennent sur 2 octets.

    Schéma mental
    -------------
        a (64, 128)  @  b (128, 32)  ->  (64, 32)   dans les trois dtypes

            float32  : 4 octets/élément   erreur relative de référence = 0
            float16  : 2 octets/élément   erreur relative ~3e-4   (< 1e-3)
            bfloat16 : 2 octets/élément   erreur relative ~3e-3   (< 1e-2)

        erreur relative = || resultat.float() - resultat_fp32 || / || resultat_fp32 ||

    Ce que ce test vérifie
    ----------------------
    1. les trois matmuls rendent la même shape (64, 32), sur `"cuda"` ;
    2. le dtype de sortie suit le dtype d'entrée : float32, float16, bfloat16 ;
    3. la demi-précision coûte 2 octets par élément contre 4 en FP32, soit deux fois moins
       de mémoire pour le même résultat ;
    4. l'erreur relative reste sous 1e-3 en FP16 et sous 1e-2 en BF16, et BF16 perd plus
       que FP16 (8 bits de mantisse contre 11).

    API à faire émerger (cible proposée : `src/inference_lab/benchmarks/matmul.py`)
    -----------------------------------------------------------------------------
        def matmul_in_dtype(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype)
            -> torch.Tensor: ...

        La roadmap indique « benchmark » sans chemin : `benchmarks/` accueille déjà le
        timing (6.2 / 6.3), ce helper y rejoint la mesure des matmuls.

    Indice : convertis les deux opérandes avec `.to(dtype)` avant le produit, et renvoie le
    résultat dans ce dtype (sans le repromouvoir en float32). Pour l'erreur, remonte en
    float32 avec `.float()` puis utilise `torch.linalg.norm` ou `.norm()`. Pièges : la
    norme d'un tenseur FP16 peut déborder, calcule-la toujours en float32 ; et
    `torch.testing.assert_close` refuse de comparer deux dtypes différents, il faut convertir
    explicitement.
    """

    pytest.skip("Roadmap TDD 6.7 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.benchmarks.matmul import matmul_in_dtype

    # Arrange — avec une seed fixée, créer sur GPU en `torch.float32` `a` de shape (64, 128)
    #           et `b` de shape (128, 32), à valeurs aléatoires centrées (échelle proche de 1,
    #           pour rester loin des limites de plage de FP16).

    # Act — calculer les trois produits `result_fp32`, `result_fp16` et `result_bf16` en
    #       passant le dtype voulu à l'API, puis mesurer les erreurs relatives
    #       `fp16_relative_error` et `bf16_relative_error` par rapport à `result_fp32`, en
    #       ramenant tout en float32 pour le calcul de la norme.

    # Assert 1 — la shape ne dépend pas du dtype
    assert result_fp32.shape == (64, 32)
    assert result_fp16.shape == (64, 32)
    assert result_bf16.shape == (64, 32)
    assert result_fp16.device.type == "cuda"

    # Assert 2 — le dtype de sortie suit le dtype d'entrée
    assert result_fp32.dtype is torch.float32
    assert result_fp16.dtype is torch.float16
    assert result_bf16.dtype is torch.bfloat16

    # Assert 3 — la demi-précision coûte deux fois moins d'octets par élément
    assert result_fp32.element_size() == 4
    assert result_fp16.element_size() == 2
    assert result_bf16.element_size() == 2

    # Assert 4 — l'erreur relative reste bornée, et BF16 perd plus que FP16
    assert fp16_relative_error < 1e-3
    assert bf16_relative_error < 1e-2
    assert bf16_relative_error > fp16_relative_error
