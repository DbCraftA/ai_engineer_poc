import pytest


@pytest.mark.tdd
@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_timer_returns_positive_elapsed_time():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_cuda_timer_returns_positive_elapsed_time`.

    Concepts à comprendre
    ---------------------
    - CUDA Event
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/benchmarks/timing.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 6.3.
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


