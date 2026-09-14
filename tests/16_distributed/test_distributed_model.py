import pytest


@pytest.mark.tdd
@pytest.mark.distributed
def test_tensor_parallel_model_matches_single_device_reference():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_tensor_parallel_model_matches_single_device_reference`.

    Concepts à comprendre
    ---------------------
    - distributed model
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    future distributed

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 16.6.
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


