"""Section 7.1 / 7.2 / 7.3 — runner de benchmark : warmup, répétitions, médiane.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/07_performance/test_benchmark_runner.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(5 appels pour 2 warmups et 3 mesures, médiane 12.0, moyenne 29.2) : c'est volontaire.
Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule que le
code testé.

RÈGLE D'OR de toute la section 7 : un test de benchmark n'asserte JAMAIS une durée
absolue (« moins de 5 ms »), qui dépend de la machine, de la charge et du dtype. On teste
la MÉCANIQUE de la mesure : combien d'appels, combien de mesures, dans quel ordre, quel
agrégat. D'où deux outils dans ce fichier : une fonction jouet instrumentée (un compteur
d'appels au lieu d'un vrai modèle) et une horloge injectable, qui rendent le protocole de
mesure déterministe et donc testable.

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
@pytest.mark.perf
def test_benchmark_executes_warmup_before_measurement():
    """Roadmap 7.1 — les itérations de warmup passent AVANT et ne sont jamais mesurées.

    Objectif d'apprentissage
    ------------------------
    Les premières exécutions d'un modèle sont systématiquement les plus lentes, et pour des
    raisons qui n'ont rien à voir avec la performance du calcul :

        - allocations paresseuses : le caching allocator CUDA doit d'abord réserver ses
          blocs, les buffers de travail (workspace) n'existent pas encore ;
        - autotuning cuBLAS / cuDNN : le premier GEMM d'une shape donnée essaie plusieurs
          algorithmes avant de retenir le meilleur ;
        - compilation et chargement : chargement du contexte CUDA et des kernels, JIT de
          `torch.compile` ou de Triton, qui peut coûter des secondes ;
        - caches froids : poids pas encore en L2/HBM, pages mémoire pas encore touchées.

    Inclure ces itérations dans la mesure fait apparaître une latence 10 à 100 fois trop
    grande et rend tout comparatif (FP16 vs BF16, avec vs sans KV cache) sans valeur. Le
    warmup s'exécute donc en dehors du chronomètre, et le runner ne renvoie que les mesures
    du régime stabilisé.

    Schéma mental
    -------------
        run_benchmark(toy_fn, warmup=2, repeats=3)

        appel #1  warmup   |  hors chronomètre
        appel #2  warmup   |  hors chronomètre
        appel #3  mesuré   ->  durée 1     ) horloge lue avant ET après
        appel #4  mesuré   ->  durée 2     ) chaque répétition
        appel #5  mesuré   ->  durée 3     )

        5 appels de la fonction, 3 durées renvoyées, 6 lectures d'horloge
        et la 1re lecture d'horloge voit déjà 2 appels effectués

    Ce que ce test vérifie
    ----------------------
    1. la fonction est appelée exactement `warmup + repeats` fois, soit 5 fois, et le run
       conserve les paramètres demandés (2 et 3) ;
    2. seules les répétitions sont mesurées : 3 durées renvoyées, pas 5 ;
    3. l'ordre est bien « warmup puis mesures » : à la première lecture d'horloge, les 2
       appels de warmup sont déjà consommés, et l'horloge est lue 2 fois par répétition ;
    4. chaque durée renvoyée est strictement positive.

    API à faire émerger (cible roadmap : `src/inference_lab/benchmarks/runner.py`)
    -----------------------------------------------------------------------------
        @dataclass(frozen=True)
        class BenchmarkRun:
            durations_seconds: list[float]
            warmup: int
            repeats: int

        def run_benchmark(
            fn: Callable[[], object],
            *,
            warmup: int = 2,
            repeats: int = 3,
            clock: Callable[[], float] = time.perf_counter,
        ) -> BenchmarkRun: ...

    Indice : la boucle est `for _ in range(warmup): fn()` PUIS, pour chaque répétition,
    `start = clock()` / `fn()` / `end = clock()` avec `durations.append(end - start)`. Le
    paramètre `clock` n'est là que pour la testabilité : il vaut `time.perf_counter` en
    production (jamais `time.time`, qui n'est pas monotone). Piège : ne remets pas les
    durées de warmup dans la liste, et n'appelle pas `clock()` une seule fois pour toute la
    boucle — tu mesurerais un total, pas des mesures individuelles.
    """

    pytest.skip("Roadmap TDD 7.1 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.benchmarks.runner import run_benchmark

    # Arrange — pas de vrai modèle ici, une fonction jouet instrumentée. Construire :
    #           `calls`, une liste vide qui sert de compteur d'appels ; `toy_fn`, une fonction
    #           sans argument qui ne fait qu'ajouter une entrée à `calls` ; `clock_ticks`, une
    #           liste vide ; `fake_clock`, une horloge factice sans argument qui, à chaque
    #           lecture, enregistre dans `clock_ticks` le nombre d'appels déjà effectués
    #           (`len(calls)`) puis renvoie une valeur avançant d'un pas fixe strictement
    #           positif, afin que chaque durée mesurée soit strictement positive.

    # Act — lancer le runner sur `toy_fn` avec 2 itérations de warmup, 3 répétitions mesurées
    #       et l'horloge factice, et garder le résultat dans `run`.

    # Assert 1 — warmup + repeats appels au total, paramètres conservés
    assert len(calls) == 5
    assert run.warmup == 2
    assert run.repeats == 3

    # Assert 2 — le warmup n'est PAS compté dans les résultats
    assert len(run.durations_seconds) == 3

    # Assert 3 — le warmup a bien lieu avant la première mesure
    assert clock_ticks[0] == 2
    assert len(clock_ticks) == 6
    assert clock_ticks[-1] == 5

    # Assert 4 — une mesure est une durée, donc strictement positive
    assert all(duration > 0.0 for duration in run.durations_seconds)


@pytest.mark.tdd
@pytest.mark.perf
def test_benchmark_collects_multiple_measurements():
    """Roadmap 7.2 — une seule mesure ne veut rien dire : le runner collecte un échantillon.

    Objectif d'apprentissage
    ------------------------
    Même après le warmup, deux exécutions identiques ne durent jamais exactement le même
    temps : ordonnancement de l'OS, fréquence du GPU qui varie (thermal/power throttling),
    interruptions, autres processus. Une mesure unique est donc un tirage aléatoire, pas une
    latence. Le runner garde l'échantillon BRUT des N répétitions ; c'est lui qui permettra
    ensuite de calculer une médiane (7.3), un écart, un p95, et de dire si un écart entre
    deux variantes est significatif ou dans le bruit.

    Ce test est aussi l'occasion de rappeler ce qu'on ne fait jamais : asserter
    « latence < 5 ms ». Un tel test passe sur ta machine et échoue en CI. On asserte le
    nombre de mesures, leur signe et leur type, pas leur valeur.

    Schéma mental
    -------------
        run_benchmark(toy_matmul, warmup=1, repeats=5)   # horloge réelle

        appels    : 1 warmup + 5 mesures = 6
        résultat  : [d1, d2, d3, d4, d5]   5 flottants, tous > 0
        agrégat   : min(durées) <= médiane <= max(durées)

        valeurs des durées : INCONNUES et machine-dépendantes, jamais assertées

    Ce que ce test vérifie
    ----------------------
    1. le cadre de mesure : un vrai calcul jouet sur des tenseurs (64, 64) en float32, appelé
       6 fois (1 warmup + 5 répétitions) ;
    2. le runner renvoie exactement `repeats` mesures, soit 5, et l'expose dans `repeats` ;
    3. chaque mesure est un `float` strictement positif : une durée nulle signalerait une
       horloge trop grossière ou un calcul optimisé hors de la boucle ;
    4. l'agrégat reste cohérent avec l'échantillon : la médiane est encadrée par le minimum
       et le maximum observés (aucune valeur absolue attendue).

    API à faire émerger (cible roadmap « runner », cible proposée :
    `src/inference_lab/benchmarks/runner.py`, module déjà visé par 7.1)
    ------------------------------------------------------------------
        run = run_benchmark(fn, warmup=1, repeats=5)
        run.durations_seconds -> list[float]
        run.median_seconds    -> float

    Indice : la valeur par défaut de `clock` (`time.perf_counter`) suffit ici, on ne
    remplace l'horloge que pour les tests déterministes. Pour que les durées soient
    franchement positives, la fonction jouet doit faire un vrai calcul : un `matmul` (64, 64)
    en float32 coûte quelques microsecondes, très au-dessus de la résolution nanoseconde de
    `perf_counter`. Piège : construis les tenseurs UNE fois dans l'Arrange, sinon tu mesures
    aussi leur allocation.
    """

    pytest.skip("Roadmap TDD 7.2 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.benchmarks.runner import run_benchmark

    # Arrange — construire `calls`, une liste vide servant de compteur d'appels, `lhs` et `rhs`,
    #           deux tenseurs `torch.float32` de shape (64, 64) alloués UNE seule fois, et
    #           `toy_matmul`, une fonction sans argument qui enregistre son appel dans `calls`
    #           puis renvoie le produit matriciel de `lhs` et `rhs`. Le calcul doit être
    #           réellement effectué pour que la durée mesurée soit non nulle.

    # Act — lancer le runner sur `toy_matmul` avec 1 itération de warmup et 5 répétitions
    #       mesurées, en gardant l'horloge réelle par défaut, et stocker le résultat dans `run`.

    # Assert 1 — le cadre du calcul jouet, et 1 warmup + 5 répétitions = 6 appels
    assert lhs.shape == (64, 64)
    assert lhs.dtype is torch.float32
    assert len(calls) == 6

    # Assert 2 — un échantillon de 5 mesures, pas une mesure unique
    assert len(run.durations_seconds) == 5
    assert run.repeats == 5

    # Assert 3 — toutes les mesures sont des durées strictement positives
    assert all(isinstance(duration, float) for duration in run.durations_seconds)
    assert all(duration > 0.0 for duration in run.durations_seconds)

    # Assert 4 — l'agrégat est encadré par l'échantillon, aucune durée absolue attendue
    assert min(run.durations_seconds) <= run.median_seconds
    assert run.median_seconds <= max(run.durations_seconds)


@pytest.mark.tdd
@pytest.mark.perf
def test_benchmark_reports_median_latency():
    """Roadmap 7.3 — la médiane résiste aux valeurs aberrantes, la moyenne non.

    Objectif d'apprentissage
    ------------------------
    Une distribution de latences n'est pas gaussienne : elle a une longue traîne à droite
    (une itération ralentie par un GC, une préemption, un pic de fréquence). Une seule
    itération 10 fois trop lente suffit à déplacer la moyenne, alors que la médiane, qui
    ne regarde que le rang des valeurs, reste sur le régime typique :

        durées  : 10, 11, 12, 13, 100     (secondes fictives)
        médiane : 12      <- valeur typique, l'aberration ne pèse qu'un rang
        moyenne : 29.2    <- plus du double, tirée par une seule mesure

    C'est pourquoi les rapports d'inférence publient une médiane (ou un p50) accompagnée
    d'un p95 : la médiane décrit le cas normal, le p95 la qualité de service. La moyenne,
    elle, ne décrit rien d'observable.

    Schéma mental
    -------------
        horloge factice rejouant des durées connues, la fonction mesurée ne fait rien

        t = 0 -> 10 -> 21 -> 33 -> 46 -> 146
                 |     |     |     |     |
        durées   10    11    12    13    100

        trié : [10, 11, 12, 13, 100], 5 valeurs -> la médiane est l'élément de rang 3 = 12

    Ce que ce test vérifie
    ----------------------
    1. le runner renvoie les 5 durées brutes, dans l'ordre de mesure et sans les trier ;
    2. la médiane vaut exactement 12.0 ;
    3. la moyenne vaut 29.2, soit plus du double de la médiane : la valeur aberrante la
       déplace, et c'est bien pour cela qu'on ne la publie pas ;
    4. sur un nombre impair de mesures, la médiane est l'une des mesures réellement
       observées, et elle reste inférieure à la moyenne sur une distribution à traîne droite.

    API à faire émerger (cible roadmap « runner », cible proposée :
    `src/inference_lab/benchmarks/runner.py`, module déjà visé par 7.1)
    ------------------------------------------------------------------
        run.durations_seconds -> list[float]     # mesures brutes, ordre de mesure
        run.median_seconds    -> float
        run.mean_seconds      -> float

    Indice : `statistics.median` et `statistics.fmean` font le travail, ne réimplémente pas
    le tri. L'horloge factice se construit à partir des sommes cumulées des durées voulues :
    le runner lisant l'horloge deux fois par répétition, il faut lui servir les valeurs
    0, 10, 10, 21, 21, 33, ... Piège : une médiane sur un nombre PAIR de mesures est la
    moyenne des deux valeurs centrales et n'appartient donc pas à l'échantillon.
    """

    pytest.skip("Roadmap TDD 7.3 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.benchmarks.runner import run_benchmark

    # Arrange — injecter des durées CONNUES au lieu de mesurer du vrai temps. La liste de
    #           référence de l'exercice est `durations = [10.0, 11.0, 12.0, 13.0, 100.0]` :
    #           quatre mesures resserrées et une valeur aberrante. Construire `replay_clock`,
    #           une horloge factice sans argument qui renvoie, lecture après lecture, les
    #           sommes cumulées de `durations` de sorte que chaque répétition du runner
    #           mesure exactement la durée correspondante ; et `noop_fn`, une fonction sans
    #           argument qui ne fait rien (le temps ne vient plus du calcul mais de l'horloge).

    # Act — lancer le runner sur `noop_fn` avec 1 itération de warmup, 5 répétitions mesurées
    #       et `replay_clock` comme horloge, puis garder le résultat dans `run`.

    # Assert 1 — les mesures brutes sont conservées telles quelles, dans l'ordre
    assert run.durations_seconds == [10.0, 11.0, 12.0, 13.0, 100.0]

    # Assert 2 — la médiane est la valeur centrale, écrite en dur
    assert run.median_seconds == 12.0

    # Assert 3 — la moyenne, elle, est emportée par la valeur aberrante
    assert run.mean_seconds == pytest.approx(29.2)
    assert run.mean_seconds > 2 * run.median_seconds

    # Assert 4 — sur 5 mesures, la médiane est une mesure réellement observée
    assert run.median_seconds in run.durations_seconds
    assert run.median_seconds < run.mean_seconds
