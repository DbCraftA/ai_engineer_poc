import pytest


@pytest.mark.tdd
@pytest.mark.distributed
def test_all_gather_reconstructs_all_shards():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_all_gather_reconstructs_all_shards`.

    Concepts à comprendre
    ---------------------
    - AllGather
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    future distributed

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 16.4.
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
@pytest.mark.distributed
def test_all_reduce_matches_single_device_reference_sum():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_all_reduce_matches_single_device_reference_sum`.

    Concepts à comprendre
    ---------------------
    - AllReduce
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    future distributed

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 16.5.
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


