"""Section 5.8 / 5.9 / 5.10 — intensité arithmétique, roofline et diagnostic du goulot.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/05_calculators/test_arithmetic_intensity.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(1 FLOP/octet en decode, point d'équilibre à 100 FLOP/octet, 1 TFLOP/s atteint sur 100 de
crête) : c'est volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec
la même formule que le code testé.

Cette dernière étape assemble les deux budgets mesurés séparément — les FLOPs de 5.6 et les
octets de 5.7 — en un unique nombre décisionnel, puis le confronte au matériel. Le matériel
de référence de ce fichier est volontairement rond : 100 TFLOP/s de crête et 1000 Go/s de
bande passante, donc un point d'équilibre à 100 FLOP/octet exactement. C'est le verdict de
5.10 qui oriente toutes les optimisations de la suite de la roadmap : inutile d'écrire un
kernel plus rapide si le GPU passe son temps à attendre la mémoire.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices. Même raison pour l'import
# du module cible : c'est ton code d'`Act` qui l'appellera, le linter le voit donc inutilisé.
# ruff: noqa: F401, F821

import pytest


@pytest.mark.tdd
def test_arithmetic_intensity_is_flops_divided_by_bytes():
    """Roadmap 5.8 — l'intensité arithmétique mesure le travail fait par octet déplacé.

    Objectif d'apprentissage
    ------------------------
    Un noyau de calcul consomme deux ressources indépendantes : des opérations flottantes et
    des octets de bande passante. Leur rapport caractérise le noyau lui-même, sans référence
    à aucun matériel :

        intensité (FLOP/octet) = FLOPs / octets_déplacés

    C'est une propriété de l'algorithme, pas de la carte : elle dit combien de calcul on
    parvient à amortir sur chaque octet chargé. Appliquée au decode d'un LLM, elle explique
    le paradoxe central du domaine — un modèle de 0,5 milliard de paramètres n'utilise
    qu'une fraction de pourcent de la puissance d'un GPU moderne. Le levier pour la faire
    monter est toujours le même : réutiliser les poids une fois chargés, donc traiter
    plusieurs tokens à la fois (prefill, ou batch de requêtes en decode).

    Schéma mental
    -------------
        Qwen2.5-0.5B, 500e6 paramètres, poids FP16 (5.1 / 5.6 / 5.7)

            decode, 1 token  : 1e9 FLOPs / 1e9 octets     =   1 FLOP/octet
            decode INT8      : 1e9 FLOPs / 0,5e9 octets   =   2 FLOP/octet
            prefill 128 tok. : 128e9 FLOPs / 1e9 octets   = 128 FLOP/octet

        les poids sont lus UNE fois quel que soit le nombre de tokens traités ensemble :
        le numérateur croît avec les tokens, le dénominateur non -> l'intensité monte

    Ce que ce test vérifie
    ----------------------
    1. le cas jouet : 1200 FLOPs pour 100 octets déplacés font 12 FLOP/octet ;
    2. le decode d'un token de Qwen2.5-0.5B en FP16 vaut exactement 1 FLOP/octet, et 2 en
       INT8 — l'ordre de grandeur de 1 à 2 FLOP/octet à batch 1, quelle que soit la
       précision ;
    3. le prefill de 128 tokens atteint 128 FLOP/octet sur les mêmes poids : traiter
       plusieurs tokens à la fois multiplie l'intensité par leur nombre ;
    4. l'intensité ne dépend pas du matériel : c'est un rapport sans référence à une carte,
       et il est invariant si l'on double simultanément FLOPs et octets.

    API à faire émerger (cible roadmap : `src/inference_lab/calculators/roofline.py`)
    -------------------------------------------------------------------------------
        def arithmetic_intensity(flops: float, bytes_moved: float) -> float: ...

    Indice : une division, mais renvoie un `float` (une division entière tronquerait 1,5 en
    1) et refuse explicitement `bytes_moved == 0`. Piège de fond : ne fais PAS entrer la
    bande passante ni la puissance crête dans cette fonction, c'est le rôle de 5.9 —
    l'intensité décrit la charge de travail, le roofline décrit la machine.
    """

    pytest.skip("Roadmap TDD 5.8 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.calculators.roofline import arithmetic_intensity

    # Arrange — que des nombres, aucun tenseur. Le cas jouet : `toy_flops` = 1200 et
    #           `toy_bytes` = 100. Puis les budgets de Qwen2.5-0.5B repris de 5.6 et 5.7 :
    #           `decode_flops` = 1e9 FLOPs pour un token, `weight_bytes_fp16` = 1e9 octets,
    #           `weight_bytes_int8` = 0,5e9 octets, et pour le prefill de 128 tokens
    #           `prefill_flops` = 128e9 FLOPs sur les mêmes `weight_bytes_fp16` octets.

    # Act — demander à l'API l'intensité arithmétique du cas jouet, du decode FP16, du decode
    #       INT8 et du prefill, et stocker les résultats dans `toy_intensity`,
    #       `decode_intensity_fp16`, `decode_intensity_int8` et `prefill_intensity`.

    # Assert 1 — le cas jouet, vérifiable de tête
    assert toy_intensity == pytest.approx(12.0, rel=1e-12)

    # Assert 2 — à batch 1, le decode plafonne à 1 ou 2 FLOP/octet selon la précision
    assert decode_intensity_fp16 == pytest.approx(1.0, rel=1e-12)
    assert decode_intensity_int8 == pytest.approx(2.0, rel=1e-12)

    # Assert 3 — traiter 128 tokens d'un coup relit les mêmes poids : intensité x128
    assert prefill_intensity == pytest.approx(128.0, rel=1e-12)
    assert prefill_intensity == pytest.approx(128 * decode_intensity_fp16, rel=1e-12)

    # Assert 4 — c'est un rapport, invariant par mise à l'échelle des deux budgets
    assert arithmetic_intensity(2 * toy_flops, 2 * toy_bytes) == pytest.approx(12.0, rel=1e-12)


@pytest.mark.tdd
def test_roofline_limit_is_minimum_of_compute_and_bandwidth_limits():
    """Roadmap 5.9 — le débit atteignable est le minimum de deux plafonds, pas leur moyenne.

    Objectif d'apprentissage
    ------------------------
    Le modèle roofline confronte l'intensité d'une charge (5.8) à deux caractéristiques de la
    carte : sa puissance crête et sa bande passante. Chacune impose un plafond de FLOP/s, et
    le débit réel ne peut dépasser le plus bas des deux :

        débit_atteignable = min(perf_crête, bande_passante x intensité)

    Le croisement des deux plafonds définit le POINT D'ÉQUILIBRE de la machine, exprimé en
    FLOP/octet :

        point_équilibre = perf_crête / bande_passante

    C'est le seul nombre à retenir d'une fiche technique GPU : il transforme une intensité en
    verdict. En dessous, la carte est bridée par la mémoire et sa puissance de calcul est
    inutilisable ; au-dessus, elle sature ses unités de calcul et la bande passante n'est
    plus le problème. Les cartes récentes ont un point d'équilibre très haut (le calcul crête
    progresse plus vite que la bande passante), ce qui rend le decode de plus en plus
    déséquilibré à chaque génération de matériel.

    Schéma mental
    -------------
        matériel de référence : crête = 100 TFLOP/s = 1e14 FLOP/s
                                bande passante = 1000 Go/s = 1e12 octets/s
                                point d'équilibre = 1e14 / 1e12 = 100 FLOP/octet

        FLOP/s
        1e14 |               ,-------------------  <- plafond calcul (crête)
             |             ,'
             |           ,'   pente = bande passante
        1e12 |      ,.-'
             +-----+---------+------------------- intensité (FLOP/octet)
                   1        100
                decode    équilibre

            intensité   1 -> min(1e14, 1e12)   = 1e12 FLOP/s   = 1 % du crête
            intensité 100 -> min(1e14, 1e14)   = 1e14 FLOP/s   = les deux plafonds coïncident
            intensité 400 -> min(1e14, 4e14)   = 1e14 FLOP/s   = plafonné par le calcul

    Ce que ce test vérifie
    ----------------------
    1. le point d'équilibre du matériel de référence vaut exactement 100 FLOP/octet ;
    2. sous le point d'équilibre, c'est la bande passante qui décide : à intensité 1, on
       n'atteint que 1e12 FLOP/s, soit 1 % de la puissance crête ;
    3. au point d'équilibre exact, les deux plafonds coïncident à 1e14 FLOP/s ;
    4. au-dessus, le `min` écrête : à intensité 400, le résultat reste 1e14 FLOP/s et non
       4e14 — augmenter l'intensité au-delà du point d'équilibre n'apporte plus rien.

    API à faire émerger (cible roadmap : `src/inference_lab/calculators/roofline.py`)
    -------------------------------------------------------------------------------
        def balance_point_flops_per_byte(
            peak_flops_per_s: float, bandwidth_bytes_per_s: float
        ) -> float: ...
        def roofline_flops_per_s(
            intensity: float, peak_flops_per_s: float, bandwidth_bytes_per_s: float
        ) -> float: ...

    Indice : littéralement `min(peak, bandwidth * intensity)`. Pièges : c'est un `min` et
    jamais une somme ni une moyenne (les deux ressources ne s'additionnent pas, elles se
    limitent) ; et garde des octets/s en entrée, pas des Go/s, sinon le point d'équilibre est
    faux d'un facteur 1e9.
    """

    pytest.skip("Roadmap TDD 5.9 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.calculators.roofline import (
        balance_point_flops_per_byte,
        roofline_flops_per_s,
    )

    # Arrange — le matériel de référence du fichier, décrit par deux nombres ronds :
    #           `peak_flops_per_s` = 1e14 (100 TFLOP/s) et `bandwidth_bytes_per_s` = 1e12
    #           (1000 Go/s). Trois intensités à tester : `decode_intensity` = 1,0 (le decode
    #           FP16 de 5.8), `balanced_intensity` = 100,0 (pile au point d'équilibre) et
    #           `compute_bound_intensity` = 400,0 (bien au-dessus).

    # Act — demander à l'API le point d'équilibre du matériel, puis le débit atteignable pour
    #       chacune des trois intensités. Stocker les résultats dans `balance_point`,
    #       `decode_throughput`, `balanced_throughput` et `compute_bound_throughput`.

    # Assert 1 — le point d'équilibre : 100 TFLOP/s pour 1000 Go/s
    assert balance_point == pytest.approx(100.0, rel=1e-12)

    # Assert 2 — sous le point d'équilibre, la bande passante commande
    assert decode_throughput == pytest.approx(1e12, rel=1e-12)
    assert decode_throughput == pytest.approx(0.01 * peak_flops_per_s, rel=1e-12)

    # Assert 3 — au point d'équilibre, les deux plafonds coïncident
    assert balanced_throughput == pytest.approx(1e14, rel=1e-12)

    # Assert 4 — au-dessus, le minimum écrête à la puissance crête
    assert compute_bound_throughput == pytest.approx(1e14, rel=1e-12)
    assert compute_bound_throughput < 4e14
    assert roofline_flops_per_s(1000.0, peak_flops_per_s, bandwidth_bytes_per_s) == pytest.approx(
        1e14, rel=1e-12
    )


@pytest.mark.tdd
def test_low_arithmetic_intensity_is_classified_as_bandwidth_limited():
    """Roadmap 5.10 — le decode d'un LLM est structurellement limité par la mémoire.

    Objectif d'apprentissage
    ------------------------
    Comparer l'intensité d'une charge au point d'équilibre de la carte donne un diagnostic
    binaire, et ce diagnostic dicte quelles optimisations valent la peine :

        intensité < point_équilibre  ->  bandwidth-bound : réduire les OCTETS
                                         (quantification, GQA, fusion de kernels)
        intensité >= point_équilibre ->  compute-bound   : réduire les FLOPs ou mieux
                                         occuper les unités de calcul

    Le decode d'un LLM tombe toujours du premier côté, et de très loin. À batch 1 son
    intensité vaut 1 à 2 FLOP/octet (5.8) parce que chaque poids chargé ne sert qu'à une
    seule multiplication-addition avant d'être jeté, contre 100 FLOP/octet nécessaires pour
    saturer la carte de référence : un facteur ~100 d'écart. Aucun kernel, aussi bien écrit
    soit-il, ne peut corriger cela — seul un changement de charge le peut (batcher les
    requêtes, ou quantifier pour transporter moins d'octets par poids). Le prefill, lui, est
    compute-bound : il traite tout le prompt sur un seul chargement des poids.

    Schéma mental
    -------------
        matériel de référence : point d'équilibre = 1e14 / 1e12 = 100 FLOP/octet

            decode FP16, batch 1  :   1 FLOP/octet  <<  100  ->  bandwidth-bound (1 % crête)
            decode INT8, batch 1  :   2 FLOP/octet  <<  100  ->  bandwidth-bound
            pile à l'équilibre    : 100 FLOP/octet   =  100  ->  compute-bound (convention >=)
            prefill 128 tokens    : 128 FLOP/octet   >  100  ->  compute-bound

        conclusion : à batch 1, changer de précision ne change PAS le diagnostic, seulement
        le facteur d'écart — il faut augmenter le nombre de tokens traités par chargement

    Ce que ce test vérifie
    ----------------------
    1. le decode FP16 à batch 1 est classé « bandwidth » ;
    2. passer en INT8 double l'intensité mais ne renverse pas le verdict : toujours
       « bandwidth » ;
    3. le prefill de 128 tokens est classé « compute », et la frontière est traitée par
       convention comme compute-bound (intensité >= point d'équilibre) ;
    4. le chiffre qui justifie le verdict : en decode, on n'exploite que 1 % de la puissance
       crête de la carte.

    API à faire émerger (cible roadmap : `src/inference_lab/calculators/roofline.py`)
    -------------------------------------------------------------------------------
        def bottleneck(
            intensity: float, peak_flops_per_s: float, bandwidth_bytes_per_s: float
        ) -> str: ...          # "bandwidth" ou "compute"

        La roadmap indique seulement « calculator » pour ce test : on propose de compléter le
        module concret de 5.8 / 5.9, qui porte déjà le point d'équilibre.

    Indice : réutilise `balance_point_flops_per_byte` au lieu de recopier la division, et
    renvoie exactement les chaînes `"bandwidth"` et `"compute"` (le test compare des chaînes,
    pas des booléens). Piège : la comparaison doit être `intensity < balance_point` pour
    « bandwidth », de sorte que l'égalité exacte tombe du côté « compute » — c'est la
    convention documentée ici, et un test doit trancher les cas frontières.
    """

    pytest.skip("Roadmap TDD 5.10 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.calculators.roofline import bottleneck, roofline_flops_per_s

    # Arrange — le même matériel de référence que 5.9 : `peak_flops_per_s` = 1e14 et
    #           `bandwidth_bytes_per_s` = 1e12, donc un point d'équilibre de 100 FLOP/octet.
    #           Quatre intensités issues de 5.8 : `decode_fp16_intensity` = 1,0,
    #           `decode_int8_intensity` = 2,0, `balanced_intensity` = 100,0 et
    #           `prefill_intensity` = 128,0.

    # Act — demander à l'API le régime limitant de chacune des quatre intensités sur ce
    #       matériel, et stocker les verdicts dans `decode_fp16_verdict`,
    #       `decode_int8_verdict`, `balanced_verdict` et `prefill_verdict`.

    # Assert 1 — à batch 1, en FP16, la mémoire est le goulot
    assert decode_fp16_verdict == "bandwidth"

    # Assert 2 — quantifier double l'intensité mais ne renverse pas le diagnostic
    assert decode_int8_verdict == "bandwidth"
    assert decode_int8_intensity < 100.0

    # Assert 3 — le prefill sature le calcul, et la frontière est compute-bound par convention
    assert prefill_verdict == "compute"
    assert balanced_verdict == "compute"
    assert bottleneck(99.0, peak_flops_per_s, bandwidth_bytes_per_s) == "bandwidth"

    # Assert 4 — le chiffre derrière le verdict : 1 % de la puissance crête exploitée
    assert roofline_flops_per_s(
        decode_fp16_intensity, peak_flops_per_s, bandwidth_bytes_per_s
    ) == pytest.approx(0.01 * peak_flops_per_s, rel=1e-12)
