"""Section 8.4 — une opération fusionnée doit rendre le même résultat que la version naïve.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/08_optimized/test_fusion_reference.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
((1, 4, 16), zéros exacts pour une porte nulle) : c'est volontaire. Un test doit énoncer la
vérité attendue, pas la recalculer avec la même formule que le code testé.

Dimensions constantes, reprises du SwiGLU de la section 2.7 : batch=1, seq=4, hidden=8,
intermediate=16. Une fusion se juge sur DEUX critères indépendants : la correction, qui est
un test (ce fichier), et le gain de bande passante, qui est un benchmark (section 7). Ce test
est le garde-fou de toute la suite : chaque kernel fusionné, en PyTorch puis en Triton
(section 9), devra prouver qu'il reproduit la référence non fusionnée.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_fused_operation_matches_unfused_reference():
    """Roadmap 8.4 — fusionner économise des allers-retours en mémoire, pas des calculs.

    Objectif d'apprentissage
    ------------------------
    L'activation SwiGLU s'écrit `silu(gate) * up`. Écrite naïvement, elle coûte deux passes :
    la première lit `gate` et écrit un tenseur temporaire `silu(gate)`, la seconde relit ce
    temporaire et `up` pour écrire le résultat. Le temporaire fait exactement la taille de la
    sortie, et sur GPU comme sur CPU c'est le trafic mémoire — pas les multiplications — qui
    plafonne ce genre d'opération élément par élément (elle est *memory-bound*). Fusionner,
    c'est faire les deux étapes en une seule lecture de `gate` et `up` et une seule écriture,
    donc supprimer une écriture et une lecture d'un tenseur (1, 4, 16). Sur Qwen2.5-0.5B, le
    MLP dominé par SwiGLU représente les deux tiers des paramètres : c'est là que la fusion
    paie. Le point que ce test grave dans le marbre : la fusion est un changement d'ORDONNANCEMENT
    des accès mémoire, elle ne doit rien changer aux valeurs, et elle n'améliore pas la précision.

    Schéma mental
    -------------
        gate (batch=1, seq=4, intermediate=16)      up (1, 4, 16)

        non fusionné : tmp = silu(gate)   ->  écrit 1x16x4 valeurs
                       out = tmp * up     ->  relit tmp, relit up, écrit out
        fusionné     : out = fused_swiglu(gate, up)  ->  lit gate + up, écrit out (1 passe)

        trafic mémoire : 5 tenseurs traversés contre 3, pour le MÊME résultat (1, 4, 16)

        rappel : silu(x) = x * sigmoid(x), donc silu(0) = 0 exactement, dans les deux versions.

    Ce que ce test vérifie
    ----------------------
    1. la sortie fusionnée a la shape (1, 4, 16) et le dtype de l'entrée : la fusion n'est
       pas une conversion de précision déguisée ;
    2. elle vaut la référence non fusionnée écrite en deux étapes explicites, à la tolérance
       fp32 rtol=1e-5 / atol=1e-6 ;
    3. cas repères exacts : une porte nulle ou une voie `up` nulle donnent des zéros, et la
       définition `silu(x) = x * sigmoid(x)` est respectée ;
    4. en bfloat16, la fusion reste fidèle à la référence non fusionnée CALCULÉE EN BF16
       (tolérance 1e-2) et conserve le dtype bf16 : fusionner ne récupère pas la précision
       perdue par la précision réduite, et n'en perd pas davantage.

    API à faire émerger (cible roadmap : `src/inference_lab/kernels/pytorch/`, cible proposée
    `src/inference_lab/kernels/pytorch/fused_swiglu.py`)
    ------------------------------------------------------------------------------------------
        def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor: ...

    Indice : la référence non fusionnée est `torch.nn.functional.silu(gate)` puis
    `tmp * up`, en deux instructions séparées ; côté fusionné, une seule expression suffit
    en PyTorch, et le vrai kernel fusionné viendra en Triton (section 9). Pièges : `silu_()`
    ou `mul_()` écriraient dans `gate`/`up` et casseraient la comparaison avec la référence ;
    et `silu(gate * up)` n'est PAS `silu(gate) * up` — la porte seule traverse l'activation.
    """

    pytest.skip("Roadmap TDD 8.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.kernels.pytorch.fused_swiglu import fused_swiglu

    # Arrange — batch=1, seq=4, intermediate=16.
    #           `gate` et `up` : deux tenseurs (1, 4, 16) en `torch.float32`, distincts, non
    #           constants, contenant des valeurs positives ET négatives (SiLU n'est pas
    #           symétrique), déterministes (seed fixée), de magnitude proche de 1.
    #           `zeros` : un tenseur (1, 4, 16) de zéros, même dtype.
    #           `gate_bf16` et `up_bf16` : les conversions de `gate` et `up` en
    #           `torch.bfloat16`.

    # Act — calculer `fused = fused_swiglu(gate, up)`, la référence non fusionnée en DEUX
    #       instructions séparées (`activated = torch.nn.functional.silu(gate)` puis
    #       `unfused = activated * up`), la même référence en bf16 (`unfused_bf16`), et
    #       `fused_bf16 = fused_swiglu(gate_bf16, up_bf16)`.

    # Assert 1 — shape et dtype préservés : ni reshape ni changement de précision
    assert fused.shape == (1, 4, 16)
    assert fused.dtype is torch.float32
    assert fused.shape == unfused.shape

    # Assert 2 — même résultat que la version non fusionnée
    torch.testing.assert_close(fused, unfused, rtol=1e-5, atol=1e-6)

    # Assert 3 — cas repères exacts et définition de SiLU
    torch.testing.assert_close(fused_swiglu(zeros, up), zeros, rtol=0.0, atol=0.0)
    torch.testing.assert_close(fused_swiglu(gate, zeros), zeros, rtol=0.0, atol=0.0)
    torch.testing.assert_close(fused, gate * torch.sigmoid(gate) * up, rtol=1e-5, atol=1e-6)

    # Assert 4 — en bf16 la fusion suit la référence bf16, sans changer le dtype
    assert fused_bf16.dtype is torch.bfloat16
    torch.testing.assert_close(fused_bf16, unfused_bf16, rtol=1e-2, atol=1e-2)
