"""Section 5.1 — mémoire des paramètres d'un modèle en fonction du dtype.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/05_calculators/test_model_memory.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(2 000 000 000 octets en FP32, 1 000 000 000 en FP16, 500 000 000 en INT8) : c'est
volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même
formule que le code testé.

Toute la section 5 est analytique : aucun tenseur n'est alloué, on manipule des formules
fermées dont on peut vérifier le résultat à la main. Cette première brique reprend la
formule de la section 1.9 (`mémoire = nombre d'éléments x octets par élément`) et
l'applique aux ~0,5 milliard de paramètres de Qwen2.5-0.5B : c'est le plancher mémoire
du modèle, celui qui décide si les poids tiennent dans la VRAM disponible.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_parameter_memory_is_parameter_count_times_dtype_size():
    """Roadmap 5.1 — la mémoire des poids, c'est un comptage de paramètres fois un dtype.

    Objectif d'apprentissage
    ------------------------
    Un modèle n'a pas « une taille » : il a une taille PAR précision. Le nombre de
    paramètres est fixé par l'architecture, mais le nombre d'octets dépend du dtype de
    stockage :

        mémoire_poids (octets) = nombre_de_paramètres x octets_par_élément(dtype)

    C'est le premier calcul à faire avant de charger un modèle : Qwen2.5-0.5B tient en
    2 Go en FP32, 1 Go en FP16, 0,5 Go en INT8. Ce même nombre revient en 5.7 comme
    volume d'octets à relire depuis la HBM à CHAQUE étape de decode, donc il pilote
    aussi la vitesse de génération, pas seulement l'occupation VRAM.

    Schéma mental
    -------------
        Qwen2.5-0.5B, arrondi à 500 000 000 paramètres

            FP32 (4 octets) : 500e6 x 4 = 2 000 000 000 octets  (~1,86 GiB)
            FP16 (2 octets) : 500e6 x 2 = 1 000 000 000 octets  (~0,93 GiB)
            INT8 (1 octet)  : 500e6 x 1 =   500 000 000 octets  (~0,47 GiB)

        même architecture, mêmes shapes, mémoire divisée par 2 puis par 4

    Ce que ce test vérifie
    ----------------------
    1. les trois valeurs attendues, calculées à la main, pour FP32, FP16 et INT8 ;
    2. les rapports exacts entre précisions : FP32 = 2 x FP16 = 4 x INT8, et BF16 coûte
       autant que FP16 (2 octets, seule la répartition des bits change) ;
    3. la conversion en gibioctets, l'unité dans laquelle on lit une VRAM : 1 GiB vaut
       2**30 octets, donc 1 000 000 000 octets font ~0,9313 GiB (et non 1 GiB) ;
    4. la linéarité de la formule : deux fois plus de paramètres à dtype constant, deux
       fois plus d'octets.

    API à faire émerger (cible roadmap : `src/inference_lab/calculators/model_memory.py`)
    ------------------------------------------------------------------------------------
        def parameter_memory_bytes(num_parameters: int, dtype: torch.dtype) -> int: ...
        def bytes_to_gib(num_bytes: int) -> float: ...

    Indice : réutilise `bytes_per_element` de la section 1.8 plutôt que de recopier une
    table de dtypes. Pièges : renvoie un `int` en octets (pas un float, pas des mégaoctets),
    et n'écris pas `/ 1e9` pour la conversion en GiB — c'est `/ 2**30`, l'écart de 7 %
    entre Go et GiB est exactement ce qui fait déborder une VRAM annoncée « 24 Go ».
    """

    pytest.skip("Roadmap TDD 5.1 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.calculators.model_memory import bytes_to_gib, parameter_memory_bytes

    # Arrange — pas de tenseur à allouer, seulement la configuration de référence du projet :
    #           `num_parameters`, le nombre de paramètres de Qwen2.5-0.5B arrondi à
    #           500 000 000 (0,5e9), en `int`. Prévoir aussi `num_parameters_doubled`, le
    #           double exact du précédent, pour l'assert 4.

    # Act — demander à l'API la mémoire des poids pour ce nombre de paramètres en
    #       `torch.float32`, `torch.float16` puis `torch.int8`, et stocker les résultats
    #       dans `fp32_bytes`, `fp16_bytes` et `int8_bytes`.

    # Assert 1 — les trois valeurs attendues, écrites en dur
    assert fp32_bytes == 2_000_000_000
    assert fp16_bytes == 1_000_000_000
    assert int8_bytes == 500_000_000

    # Assert 2 — les rapports entre précisions, à architecture identique
    assert fp32_bytes == 2 * fp16_bytes
    assert fp32_bytes == 4 * int8_bytes
    assert parameter_memory_bytes(num_parameters, torch.bfloat16) == fp16_bytes

    # Assert 3 — un gibioctet vaut 2**30 octets, pas 1e9
    assert bytes_to_gib(fp16_bytes) == pytest.approx(0.9313, rel=1e-4)
    assert bytes_to_gib(fp32_bytes) == pytest.approx(1.8626, rel=1e-4)

    # Assert 4 — la formule est linéaire en nombre de paramètres
    assert parameter_memory_bytes(num_parameters_doubled, torch.float16) == 2_000_000_000
