import pytest


@pytest.mark.tdd
def test_block_allocator_assigns_fixed_size_kv_blocks():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_block_allocator_assigns_fixed_size_kv_blocks`.

    Concepts à comprendre
    ---------------------
    - KV blocks
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    memory manager

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 14.4.
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
def test_logical_kv_blocks_map_to_physical_blocks():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_logical_kv_blocks_map_to_physical_blocks`.

    Concepts à comprendre
    ---------------------
    - block table
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    paged cache

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 14.5.
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


