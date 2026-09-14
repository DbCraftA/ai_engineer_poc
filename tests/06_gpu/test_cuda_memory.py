import pytest


@pytest.mark.tdd
@pytest.mark.gpu
@pytest.mark.cuda
def test_gpu_allocation_increases_allocated_memory():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_gpu_allocation_increases_allocated_memory`.

    Concepts à comprendre
    ---------------------
    - allocation VRAM
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/profiling/memory.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 6.4.
    Lorsque la section sera activée, elle sera remplacée par un vrai Arrange / Act / Assert.

    Critère de réussite
    -------------------
    Le test doit d'abord échouer en RED pour une raison pertinente, puis passer en GREEN
    après l'implémentation minimale dans src/.

    TDD
    ---
    1. supprimer pytest.skip()
    2. écrire l'assertion attendue
    3. obtenir RED
    4. implémenter le minimum dans src/
    5. obtenir GREEN
    6. refactorer sans changer le comportement
    """
    pytest.skip("Roadmap TDD — section pas encore activée")


@pytest.mark.tdd
@pytest.mark.gpu
@pytest.mark.cuda
def test_peak_memory_can_be_recorded():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_peak_memory_can_be_recorded`.

    Concepts à comprendre
    ---------------------
    - allocation VRAM
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/profiling/memory.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 6.4.
    Lorsque la section sera activée, elle sera remplacée par un vrai Arrange / Act / Assert.

    Critère de réussite
    -------------------
    Le test doit d'abord échouer en RED pour une raison pertinente, puis passer en GREEN
    après l'implémentation minimale dans src/.

    TDD
    ---
    1. supprimer pytest.skip()
    2. écrire l'assertion attendue
    3. obtenir RED
    4. implémenter le minimum dans src/
    5. obtenir GREEN
    6. refactorer sans changer le comportement
    """
    pytest.skip("Roadmap TDD — section pas encore activée")


