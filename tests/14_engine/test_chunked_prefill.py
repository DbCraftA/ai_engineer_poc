import pytest


@pytest.mark.tdd
def test_long_prefill_can_be_split_into_multiple_chunks():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_long_prefill_can_be_split_into_multiple_chunks`.

    Concepts à comprendre
    ---------------------
    - chunked prefill
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    scheduler

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 14.8.
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


