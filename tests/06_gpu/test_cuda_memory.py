"""Section 6.4 — mémoire GPU : ce qui est alloué maintenant, et le pic atteint.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/06_gpu/test_cuda_memory.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(4 194 304 octets pour 1024 x 1024 en float32, 8 388 608 pour 2048 x 1024) : c'est
volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même
formule que le code testé.

C'est la mesure qui valide les calculs théoriques de la section 5 : la formule
`numel x octets_par_élément` prédit la mémoire, `torch.cuda.memory_allocated()` la
constate. Et comme l'allocateur de PyTorch met les blocs en cache et les arrondit, les
comparaisons se font avec `>=`, jamais avec `==` sur le delta observé.

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
def test_gpu_allocation_increases_allocated_memory():
    """Roadmap 6.4 — allouer un tenseur en VRAM augmente la mémoire allouée d'autant.

    Objectif d'apprentissage
    ------------------------
    Sur GPU, la mémoire est la première ressource à saturer : les poids de Qwen2.5-0.5B en
    FP16 tiennent dans 1 Gio, mais le KV cache d'un long contexte peut les dépasser. Savoir
    lire l'occupation réelle est donc indispensable pour diagnostiquer un `CUDA out of
    memory` et pour vérifier que les estimations de la section 5 ne mentent pas. Point clé :
    l'allocateur de PyTorch garde des blocs en cache et arrondit les tailles, donc la
    mémoire *allouée* croît d'AU MOINS la taille logique du tenseur, parfois plus.

    Schéma mental
    -------------
        (1024, 1024) float32 -> 1 048 576 éléments x 4 octets = 4 194 304 octets (4 Mio)

        memory_allocated()      avant :  A
        allocation du tenseur   après :  A + delta,  avec delta >= 4 194 304
        `del tenseur`                 :  retour à A  (le bloc revient à l'allocateur)

        memory_reserved() ne redescend pas : le cache garde le bloc pour la prochaine fois

    Ce que ce test vérifie
    ----------------------
    1. la taille logique du tenseur vaut exactement 4 194 304 octets ;
    2. la mémoire allouée augmente d'au moins cette taille (`>=`, l'allocateur arrondit) ;
    3. l'API est cohérente avec l'oracle `torch.cuda.memory_allocated()` et rend un `int` ;
    4. libérer le tenseur rend la mémoire à l'allocateur : on retrouve la valeur de départ.

    API à faire émerger (cible roadmap : `src/inference_lab/profiling/memory.py`)
    ---------------------------------------------------------------------------
        def allocated_bytes(device: torch.device | int | None = None) -> int: ...

    Indice : `torch.cuda.memory_allocated(device)` donne la réponse ; l'intérêt du module
    est de nommer le concept et de le rendre réutilisable par les benchmarks. Piège : ne
    confonds pas `memory_allocated()` (ce que tes tenseurs occupent) et `memory_reserved()`
    (ce que l'allocateur a réservé auprès du driver, qui ne diminue pas après un `del`).
    """

    pytest.skip("Roadmap TDD 6.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.profiling.memory import allocated_bytes

    # Arrange — relever `bytes_before`, la mémoire allouée avant toute allocation de ce test.
    #           Prévoir `tensor_bytes`, la taille logique d'un tenseur (1024, 1024) en
    #           `torch.float32` calculée avec `numel()` et `element_size()`.

    # Act — allouer ce tenseur directement sur GPU, relever `bytes_after`, puis le libérer
    #       (supprimer la référence Python) et relever `bytes_after_release`. Garder
    #       `tensor_bytes` sous la main : le tenseur, lui, ne doit plus exister à l'assert.

    # Assert 1 — la taille logique attendue, calculée à la main
    assert tensor_bytes == 4194304

    # Assert 2 — l'allocateur peut arrondir vers le haut, jamais vers le bas
    assert bytes_after - bytes_before >= tensor_bytes

    # Assert 3 — cohérence avec l'oracle PyTorch, et type de retour
    assert isinstance(bytes_after, int)
    assert bytes_after_release == torch.cuda.memory_allocated()

    # Assert 4 — libérer le tenseur rend la mémoire à l'allocateur
    assert bytes_after_release == bytes_before


@pytest.mark.tdd
@pytest.mark.gpu
@pytest.mark.cuda
def test_peak_memory_can_be_recorded():
    """Roadmap 6.4 — le pic mémoire garde la trace d'un tenseur déjà libéré.

    Objectif d'apprentissage
    ------------------------
    Ce qui fait tomber un serveur d'inférence, ce n'est pas la mémoire moyenne, c'est le
    PIC : un buffer d'attention intermédiaire de quelques millisecondes suffit à déclencher
    un `CUDA out of memory`. L'occupation instantanée ne le voit pas, puisque le buffer est
    déjà libéré quand on regarde. `max_memory_allocated()` mémorise ce maximum depuis le
    dernier `reset_peak_memory_stats()` : c'est l'instrument à encadrer autour d'un prefill
    ou d'un decode pour dimensionner honnêtement le matériel.

    Schéma mental
    -------------
        reset du pic                       pic = occupation courante
        alloc big   (2048, 1024) float32   = 8 388 608 octets  -> pic monte à >= 8 Mio
        del  big                           occupation redescend, PIC RESTE à 8 Mio
        alloc small (1024, 1024) float32   = 4 194 304 octets  -> occupation = 4 Mio

        pic (>= 8 Mio)  >  occupation courante (4 Mio)
        après un nouveau reset : pic == occupation courante

    Ce que ce test vérifie
    ----------------------
    1. le pic retient les 8 388 608 octets du gros tenseur, pourtant déjà libéré ;
    2. le pic est supérieur ou égal à l'occupation courante, qui vaut au moins 4 194 304 ;
    3. cohérence avec l'oracle `torch.cuda.max_memory_allocated()` ;
    4. après remise à zéro, le pic retombe exactement sur l'occupation courante.

    API à faire émerger (cible roadmap : `src/inference_lab/profiling/memory.py`)
    ---------------------------------------------------------------------------
        def peak_allocated_bytes(device: torch.device | int | None = None) -> int: ...
        def reset_peak_memory(device: torch.device | int | None = None) -> None: ...

    Indice : `torch.cuda.max_memory_allocated()` et `torch.cuda.reset_peak_memory_stats()`.
    Piège : sans reset au début, tu mesures le pic de TOUT le processus depuis son
    démarrage (imports, warmups, autres tests) et la valeur ne veut plus rien dire.
    """

    pytest.skip("Roadmap TDD 6.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.profiling.memory import (
        allocated_bytes,
        peak_allocated_bytes,
        reset_peak_memory,
    )

    # Arrange — remettre le compteur de pic à zéro pour partir d'une base propre, puis prévoir
    #           deux tailles : un gros tenseur (2048, 1024) en `torch.float32` et un petit
    #           (1024, 1024) dans le même dtype.

    # Act — allouer le gros tenseur sur GPU, le libérer, allouer ensuite le petit sur GPU,
    #       puis relever `peak_bytes` (le pic mémorisé), `current_bytes` (l'occupation
    #       courante) et `peak_bytes_from_torch` (la même valeur lue directement sur
    #       `torch.cuda`, avant toute remise à zéro : c'est l'oracle). Remettre enfin le pic
    #       à zéro et relever `peak_after_reset`.

    # Assert 1 — le pic garde la trace du gros tenseur, déjà libéré
    assert peak_bytes >= 8388608

    # Assert 2 — le pic domine l'occupation courante, qui ne contient plus que le petit
    assert current_bytes >= 4194304
    assert peak_bytes >= current_bytes

    # Assert 3 — cohérence avec l'oracle PyTorch, relevé avant la remise à zéro
    assert isinstance(peak_bytes, int)
    assert peak_bytes == peak_bytes_from_torch

    # Assert 4 — après remise à zéro, le pic repart de l'occupation courante
    assert peak_after_reset == current_bytes
