"""Section 6.2 — asynchronisme CUDA : mesurer un temps GPU sans se mentir.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/06_gpu/test_cuda_synchronization.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(durées strictement positives, marge de 1e-3 s, file d'attente vide) : c'est volontaire.
Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule que le
code testé.

C'est LE piège de tout le chapitre 7 (benchmarks) : un appel GPU rend la main
immédiatement, avant même que le kernel ait commencé. Un chronomètre CPU naïf mesure donc
le temps de *soumission*, pas le temps de calcul — on croit avoir un modèle 100x plus
rapide alors qu'on a seulement mesuré une file d'attente. La correction s'énonce en une
ligne : synchroniser avant de lire l'horloge, ou mesurer avec des `torch.cuda.Event`.

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
def test_cuda_timing_requires_synchronization_or_cuda_events():
    """Roadmap 6.2 — sans synchronisation, on chronomètre la soumission, pas le calcul.

    Objectif d'apprentissage
    ------------------------
    Le CPU pousse les kernels dans un *stream* CUDA et continue son chemin : `a @ b` sur
    GPU revient en quelques dizaines de microsecondes même si le kernel en prend trente
    millisecondes. Toute mesure de latence de prefill, de temps par token en decode ou de
    bande passante mémoire est fausse tant qu'on n'a pas fermé cette fenêtre. Le helper de
    timing du dépôt doit donc porter la synchronisation lui-même, pour qu'aucun benchmark
    de la roadmap n'ait à s'en souvenir.

    Schéma mental
    -------------
        CPU : perf_counter() --lance matmul (4096x4096)--> perf_counter()   ~0.1 ms  (faux)
        GPU :                 [========== kernel ~30 ms ==========]

        CPU : perf_counter() --lance--> synchronize() --attend la fin--> perf_counter()
                                                                        ~30 ms  (vrai)

        durée_sans_sync  <=  durée_avec_sync,  et  durée_avec_sync >= temps GPU des Event

    Ce que ce test vérifie
    ----------------------
    1. les deux mesures sont des durées réelles : des `float` strictement positifs ;
    2. la mesure synchronisée ne sous-estime jamais la mesure non synchronisée (marge de
       1e-3 s pour absorber le bruit d'ordonnancement) ;
    3. la mesure synchronisée couvre au moins le temps GPU mesuré par des `torch.cuda.Event`,
       qui sert d'oracle ;
    4. le helper laisse le stream vide quand on lui demande de synchroniser : il a bien
       attendu la fin du kernel, il ne s'est pas contenté de lire l'horloge.

    API à faire émerger (cible roadmap : `src/inference_lab/benchmarks/timing.py`)
    ----------------------------------------------------------------------------
        def time_cuda_call(fn: Callable[[], object], synchronize: bool = True) -> float: ...

        Renvoie une durée en SECONDES. Avec `synchronize=True`, appelle
        `torch.cuda.synchronize()` avant de lire l'horloge de fin.

    Indice : `time.perf_counter()` pour l'horloge CPU, `torch.cuda.synchronize()` pour
    attendre le GPU, `torch.cuda.default_stream().query()` pour savoir si la file est
    vide. Piège : mesure toujours APRÈS un warmup (le premier appel paie le chargement des
    kernels et l'initialisation du contexte CUDA), et synchronise aussi AVANT de démarrer
    le chronomètre, sinon tu comptabilises le travail en retard de la mesure précédente.
    """

    pytest.skip("Roadmap TDD 6.2 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.benchmarks.timing import time_cuda_call

    # Arrange — créer un tenseur carré `weights` en `torch.float32` sur GPU, assez grand pour
    #           que le kernel dure plusieurs millisecondes (4096 x 4096 convient), puis
    #           `heavy_matmul`, un callable sans argument qui fait `weights @ weights`.
    #           Faire un warmup (quelques appels) suivi d'une synchronisation complète, pour
    #           que la première mesure ne paie pas l'initialisation du contexte CUDA.

    # Act — mesurer `heavy_matmul` sans synchronisation dans `elapsed_without_sync`, puis avec
    #       synchronisation dans `elapsed_with_sync`, en synchronisant entre les deux mesures.
    #       Mesurer aussi le même appel avec une paire de `torch.cuda.Event(enable_timing=True)`
    #       et convertir le résultat (des millisecondes) en secondes dans `gpu_seconds`.
    #       Relever enfin `torch.cuda.default_stream().query()` juste après un appel
    #       synchronisé du helper, dans `stream_is_idle_after_sync`.

    # Assert 1 — les deux mesures sont des durées réelles
    assert isinstance(elapsed_without_sync, float)
    assert isinstance(elapsed_with_sync, float)
    assert elapsed_without_sync > 0.0
    assert elapsed_with_sync > 0.0

    # Assert 2 — la mesure synchronisée ne sous-estime jamais l'autre
    assert elapsed_with_sync >= elapsed_without_sync - 1e-3

    # Assert 3 — elle englobe le temps GPU mesuré par les Event, notre oracle
    assert gpu_seconds > 0.0
    assert elapsed_with_sync >= gpu_seconds - 1e-3

    # Assert 4 — le helper a réellement attendu la fin du kernel
    assert stream_is_idle_after_sync is True
