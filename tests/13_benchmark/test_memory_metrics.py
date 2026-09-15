"""Section 13.4 — mémoire : pic d'occupation du device dans un rapport de benchmark.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/13_benchmark/test_memory_metrics.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(1024 x 1024 flottants FP32 = 4 194 304 octets = 4,0 MiB) : c'est volontaire. Un test doit
énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

RÈGLE D'OR, reprise de la section 7 : on n'asserte jamais une durée. Ici on n'asserte pas
davantage un pic mémoire absolu de la machine : on vérifie la MÉCANIQUE de la mesure (remise à
zéro du compteur avant l'exécution, unité de conversion, chemin CPU) et une borne INFÉRIEURE
déduite de la taille exacte du tenseur jouet, qui, elle, est connue.

Portabilité assumée : le pic mémoire d'un device n'existe que sur CUDA
(`torch.cuda.max_memory_allocated()`). Sur CPU, PyTorch n'expose aucun compteur équivalent, et
lire la RSS du processus mesurerait l'interpréteur entier, pas la charge : le rapport porte
donc `peak_bytes = None` sur CPU. C'est un « non mesurable » explicite, pas un zéro trompeur —
un 0 se moyennerait et se tracerait dans une courbe comme une vraie valeur. Le test reste donc
exécutable sans GPU : les assertions CUDA sont gardées par `torch.cuda.is_available()`.

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
def test_benchmark_reports_peak_device_memory():
    """Roadmap 13.4 — un pic mémoire n'a de sens que si le compteur est remis à zéro avant.

    Objectif d'apprentissage
    ------------------------
    En inférence, la mémoire est une contrainte aussi dure que le temps : c'est elle qui fixe
    le batch maximal, la longueur de contexte admissible et donc le débit atteignable
    (sections 5.2 à 5.5 pour les budgets théoriques, 14.4 pour les blocs de KV cache). Ce qui
    compte n'est pas la mémoire occupée à la fin, mais le PIC atteint pendant l'exécution :
    c'est lui qui déclenche l'`OutOfMemoryError`. Les activations transitoires d'un prefill
    long, par exemple, disparaissent avant la fin de la passe tout en ayant dimensionné le
    besoin réel.

    PyTorch tient ce compteur pour CUDA :

        torch.cuda.reset_peak_memory_stats()   remet le maximum au niveau ACTUELLEMENT alloué ;
        torch.cuda.max_memory_allocated()      renvoie le maximum atteint depuis cette remise.

    Sans la remise à zéro, on hérite du pic de tous les tests précédents dans le même
    processus : la mesure devient un maximum historique et non le coût de la charge observée.
    Deuxième subtilité : `max_memory_allocated` compte les tenseurs demandés par PyTorch, pas
    la mémoire réservée par le *caching allocator*
    (`max_memory_reserved`), toujours supérieure ; et il faut `torch.cuda.synchronize()` avant
    la lecture, les allocations étant asynchrones (section 6.2).

    Schéma mental
    -------------
        tenseur jouet : (1024, 1024) float32
            1024 x 1024 = 1 048 576 éléments x 4 octets = 4 194 304 octets = 4,0 MiB

        mesure sur CUDA                        mesure sur CPU
        ---------------                        --------------
        reset du pic       -> 0                pas de compteur exposé
        allocation         -> 4 MiB            peak_bytes = None
        pic relevé         >= 4 194 304        peak_mebibytes = None
        deuxième mesure    == première         (mécanique identique, valeur non mesurable)

    Ce que ce test vérifie
    ----------------------
    1. le cadre chiffré : le tenseur jouet occupe exactement 4 194 304 octets, et la
       conversion en MiB (division par 1024 x 1024, jamais par 1e6) donne 4,0 ;
    2. le chemin CPU documenté : le rapport se construit sans GPU, expose le device demandé et
       porte `peak_bytes is None` — un « non mesurable » explicite plutôt qu'un faux zéro ;
    3. sur une machine CUDA, le pic relevé est un entier au moins égal aux 4 194 304 octets du
       tenseur jouet, et cohérent avec sa conversion en MiB ;
    4. sur cette même machine, deux mesures successives de la MÊME charge rendent le même pic :
       preuve que le compteur est bien remis à zéro avant chaque mesure et que les pics ne
       s'accumulent pas d'une mesure à l'autre.

    API à faire émerger (la roadmap dit seulement « metrics », cible proposée :
    `src/inference_lab/metrics/memory.py`)
    -------------------------------------------------------------------------
        @dataclass(frozen=True)
        class DeviceMemoryUsage:
            device: str
            peak_bytes: int | None
            peak_mebibytes: float | None

        def bytes_to_mebibytes(num_bytes: int) -> float: ...

        def measure_peak_memory(
            fn: Callable[[], object], *, device: str = "cpu"
        ) -> DeviceMemoryUsage: ...

    Indice : `measure_peak_memory` doit, dans cet ordre, synchroniser, appeler
    `torch.cuda.reset_peak_memory_stats()`, exécuter `fn()`, synchroniser à nouveau, puis lire
    `torch.cuda.max_memory_allocated()`. Sur un device non CUDA, elle exécute simplement `fn()`
    et renvoie `peak_bytes=None`. Pièges : la charge jouet ne doit PAS garder de référence sur
    le tenseur alloué (sinon la mémoire reste occupée et la deuxième mesure part d'un niveau
    plus haut) ; et n'utilise pas `memory_allocated()` (instantané courant) là où il faut
    `max_memory_allocated()` (maximum atteint).
    """

    pytest.skip("Roadmap TDD 13.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.metrics.memory import bytes_to_mebibytes, measure_peak_memory

    # Arrange — une charge jouet dont la taille est connue à l'octet. Construire `toy_bytes`,
    #           le nombre d'octets d'un tenseur (1024, 1024) en `torch.float32` obtenu depuis
    #           `numel()` et `element_size()` ; puis `allocate_toy_tensor`, une fonction sans
    #           argument qui alloue un tel tenseur sur le device visé et ne renvoie AUCUNE
    #           référence vers lui, de sorte qu'il soit libéré dès la sortie. Prévoir la
    #           variante CUDA de la charge seulement si `torch.cuda.is_available()`.

    # Act — mesurer le pic mémoire de `allocate_toy_tensor` sur le device `"cpu"` et garder le
    #       rapport dans `cpu_usage`. Si et seulement si CUDA est disponible, mesurer DEUX fois
    #       de suite la même charge sur `"cuda"` et garder les rapports dans `cuda_usage` et
    #       `cuda_usage_again`.

    # Assert 1 — le cadre chiffré : 4 MiB exactement, et la conversion en base 1024
    assert toy_bytes == 4194304
    assert bytes_to_mebibytes(4194304) == 4.0
    assert bytes_to_mebibytes(toy_bytes) == pytest.approx(4.0, rel=1e-12)

    # Assert 2 — chemin CPU : mesurable partout, pic volontairement non renseigné
    assert cpu_usage.device == "cpu"
    assert cpu_usage.peak_bytes is None
    assert cpu_usage.peak_mebibytes is None

    # Assert 3 — sur CUDA, le pic couvre au moins le tenseur jouet
    if torch.cuda.is_available():
        assert isinstance(cuda_usage.peak_bytes, int)
        assert cuda_usage.peak_bytes >= 4194304
        assert cuda_usage.peak_mebibytes == pytest.approx(
            bytes_to_mebibytes(cuda_usage.peak_bytes), rel=1e-12
        )

    # Assert 4 — sur CUDA, la remise à zéro empêche les pics de s'accumuler
    if torch.cuda.is_available():
        assert cuda_usage_again.peak_bytes == cuda_usage.peak_bytes
