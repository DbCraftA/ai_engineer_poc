"""Section 5.7 — trafic mémoire : les octets relus depuis la HBM à chaque étape de decode.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/05_calculators/test_memory_traffic.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(1 000 000 000 octets par étape en FP16, 1000 tokens/s à 1000 Go/s) : c'est volontaire. Un
test doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code
testé.

Cette section fournit le DÉNOMINATEUR de l'intensité arithmétique de 5.8 : les FLOPs de 5.6
ne disent rien tout seuls, il faut les rapporter aux octets qui traversent le bus mémoire.
La quantité de calcul et la quantité de données déplacées sont deux budgets distincts, et
c'est le plus contraignant des deux qui fixe le débit réel (section 5.9).

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
def test_weight_streaming_bytes_can_be_estimated_from_parameter_size():
    """Roadmap 5.7 — générer un token oblige à relire TOUS les poids depuis la HBM.

    Objectif d'apprentissage
    ------------------------
    Un GPU ne calcule pas sur les poids stockés en HBM : il les fait d'abord transiter vers
    ses registres et sa SRAM. Or les caches d'un GPU (quelques dizaines de Mo) ne peuvent
    pas retenir 1 Go de poids d'une étape à l'autre. Conséquence : à chaque token généré,
    l'intégralité des poids retraverse le bus mémoire.

        octets_par_étape = nombre_de_paramètres x octets_par_élément(dtype)

    C'est numériquement la même expression que la mémoire des poids de 5.1, mais ce n'est
    pas la même grandeur : 5.1 mesure une OCCUPATION (payée une fois au chargement), 5.7
    mesure un DÉBIT (payé à chaque étape de decode). C'est cette relecture répétée qui rend
    le decode lent, pas le calcul : diviser ces octets par la bande passante de la carte
    donne directement le plafond de tokens par seconde, indépassable quel que soit le GPU.

    Schéma mental
    -------------
        Qwen2.5-0.5B (500 000 000 paramètres), carte à 1000 Go/s = 1e12 octets/s

            FP16 : 500e6 x 2 = 1 000 000 000 octets par étape
                   1e9 / 1e12 = 1 ms par token  ->  plafond 1000 tokens/s

            INT8 : 500e6 x 1 =   500 000 000 octets par étape
                   plafond 2000 tokens/s   (quantifier, c'est acheter du débit)

        100 tokens générés  ->  100 x 1e9 = 100 000 000 000 octets traversent le bus

    Ce que ce test vérifie
    ----------------------
    1. les octets relus par étape de decode pour les trois précisions : 1 000 000 000 en
       FP16, 2 000 000 000 en FP32, 500 000 000 en INT8 ;
    2. l'égalité numérique avec la mémoire des poids de 5.1 — même valeur, deux
       interprétations : occupation une fois, trafic à chaque étape ;
    3. la linéarité en nombre d'étapes : générer 100 tokens déplace 100 000 000 000 octets ;
    4. la conséquence pratique : à 1000 Go/s, le plafond est de 1000 tokens/s en FP16 et de
       2000 tokens/s en INT8, et ce plafond ne dépend d'aucune considération de calcul.

    API à faire émerger (cible roadmap : `src/inference_lab/calculators/memory_traffic.py`)
    --------------------------------------------------------------------------------------
        def decode_weight_bytes(num_parameters: int, dtype: torch.dtype) -> int: ...
        def bandwidth_token_ceiling(
            num_parameters: int, dtype: torch.dtype, bandwidth_bytes_per_s: float
        ) -> float: ...

    Indice : réutilise `bytes_per_element` de la section 1.8. Deux pièges. D'abord le KV
    cache : lui aussi est relu à chaque étape, mais 12 MiB sur 1024 tokens (5.2) face à 1 Go
    de poids, c'est ~1 % — l'estimation le néglige, et ce test ne l'inclut pas. Ensuite les
    unités : 1000 Go/s vaut 1e12 octets/s dans la convention constructeur (puissances de 10),
    pas 2**30 — ne mélange pas ces octets « décimaux » avec les GiB de 5.1.
    """

    pytest.skip("Roadmap TDD 5.7 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.calculators.memory_traffic import (
        bandwidth_token_ceiling,
        decode_weight_bytes,
    )
    from inference_lab.calculators.model_memory import parameter_memory_bytes

    # Arrange — aucun tenseur : `num_parameters` = 500 000 000 (Qwen2.5-0.5B arrondi),
    #           `generated_tokens` = 100 étapes de decode, et une carte de référence décrite
    #           par sa seule bande passante `bandwidth_bytes_per_s` = 1e12 octets/s
    #           (soit 1000 Go/s, un ordre de grandeur réaliste de HBM récente).

    # Act — demander à l'API les octets de poids relus par étape de decode en `torch.float16`,
    #       `torch.float32` puis `torch.int8`, et stocker les résultats dans `fp16_bytes`,
    #       `fp32_bytes` et `int8_bytes`.

    # Assert 1 — les octets relus à chaque étape, une valeur par précision
    assert fp16_bytes == 1_000_000_000
    assert fp32_bytes == 2_000_000_000
    assert int8_bytes == 500_000_000

    # Assert 2 — même nombre que l'occupation mémoire de 5.1, mais payé à chaque étape
    assert fp16_bytes == parameter_memory_bytes(num_parameters, torch.float16)

    # Assert 3 — le trafic total est linéaire en nombre de tokens générés
    assert generated_tokens * fp16_bytes == 100_000_000_000

    # Assert 4 — le plafond de débit imposé par la seule bande passante
    assert bandwidth_token_ceiling(
        num_parameters, torch.float16, bandwidth_bytes_per_s
    ) == pytest.approx(1000.0, rel=1e-9)
    assert bandwidth_token_ceiling(
        num_parameters, torch.int8, bandwidth_bytes_per_s
    ) == pytest.approx(2000.0, rel=1e-9)
