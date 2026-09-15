"""Section 6.3 — chronomètre GPU : mesurer en millisecondes avec `torch.cuda.Event`.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/06_gpu/test_cuda_timing.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(durée strictement positive, unité en millisecondes donc < 1000.0 pour un petit kernel,
facteur 10 de cohérence avec l'oracle) : c'est volontaire. Un test doit énoncer la vérité
attendue, pas la recalculer avec la même formule que le code testé.

La section 6.2 a montré POURQUOI il faut synchroniser ; celle-ci fournit l'instrument
définitif : deux `torch.cuda.Event(enable_timing=True)` enregistrés dans le stream, dont
l'écart est mesuré par le GPU lui-même, en millisecondes. C'est l'unité de toutes les
métriques du chapitre 7 (latence de prefill, temps par token en decode).

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
def test_cuda_timer_returns_positive_elapsed_time():
    """Roadmap 6.3 — un chronomètre à base d'Event rend une durée positive, en ms.

    Objectif d'apprentissage
    ------------------------
    Un `torch.cuda.Event` est un marqueur posé DANS le stream : il est daté par le GPU au
    moment où il est atteint, pas par le CPU au moment où il est soumis. `start.elapsed_time(end)`
    donne donc le temps réellement passé sur le device, en millisecondes flottantes. C'est
    la mesure que le dépôt utilisera pour comparer une attention naïve à SDPA (section 8),
    un kernel Triton à son équivalent PyTorch (section 9), ou deux backends (section 17).
    Une seule règle à retenir : l'écart n'est lisible qu'après `torch.cuda.synchronize()`.

    Schéma mental
    -------------
        stream CUDA :  [start] --- kernel --- [end]
                          |                     |
                     daté par le GPU       daté par le GPU

        start.elapsed_time(end) -> float, en MILLISECONDES (pas en secondes)

        matmul (256, 256)   -> ~0.1 ms      \\  même code, deux ordres de grandeur :
        matmul (4096, 4096) -> ~30 ms       /   la durée suit la taille du problème

    Ce que ce test vérifie
    ----------------------
    1. la durée mesurée est un `float` strictement positif ;
    2. l'unité est bien la milliseconde : un petit kernel reste très en dessous de 1000.0 ;
    3. le chronomètre discrimine deux charges : le gros matmul dure plus que le petit ;
    4. la mesure du helper est cohérente (à un facteur 10 près, on ne teste pas la
       performance) avec une paire d'`Event` posée à la main, qui sert d'oracle, et les
       deux Event sont bien complétés après synchronisation.

    API à faire émerger (cible roadmap : `src/inference_lab/benchmarks/timing.py`)
    ----------------------------------------------------------------------------
        def time_cuda_event_ms(fn: Callable[[], object], warmup: int = 1) -> float: ...

    Indice : `start = torch.cuda.Event(enable_timing=True)`, `start.record()`, l'appel,
    `end.record()`, `torch.cuda.synchronize()`, puis `start.elapsed_time(end)`. Pièges :
    sans `enable_timing=True` l'Event ne sait pas mesurer ; lire `elapsed_time` avant la
    synchronisation lève une `RuntimeError` ; et sans warmup la première mesure inclut la
    compilation / le chargement des kernels et sort dix fois trop grande.
    """

    pytest.skip("Roadmap TDD 6.3 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.benchmarks.timing import time_cuda_event_ms

    # Arrange — créer deux tenseurs carrés `small` (256 x 256) et `big` (4096 x 4096) en
    #           `torch.float32` sur GPU, puis deux callables sans argument `small_matmul` et
    #           `big_matmul` qui multiplient chacun son tenseur par lui-même. Créer aussi
    #           `start_event` et `end_event`, deux `torch.cuda.Event(enable_timing=True)`.

    # Act — chronométrer les deux callables avec l'API dans `elapsed_small_ms` et
    #       `elapsed_big_ms`, puis refaire à la main la mesure du gros matmul en encadrant
    #       l'appel par `start_event` / `end_event`, en synchronisant, et en lisant l'écart
    #       dans `reference_big_ms`.

    # Assert 1 — une durée, pas un tenseur ni None
    assert isinstance(elapsed_small_ms, float)
    assert elapsed_small_ms > 0.0
    assert elapsed_big_ms > 0.0

    # Assert 2 — l'unité est la milliseconde
    assert elapsed_small_ms < 1000.0

    # Assert 3 — le chronomètre distingue deux charges d'ordres de grandeur différents
    assert elapsed_big_ms > elapsed_small_ms

    # Assert 4 — cohérence avec l'oracle Event posé à la main
    assert reference_big_ms > 0.0
    assert elapsed_big_ms <= 10.0 * reference_big_ms
    assert elapsed_big_ms >= reference_big_ms / 10.0
    assert start_event.query() is True
    assert end_event.query() is True
