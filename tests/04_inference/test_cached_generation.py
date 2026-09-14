import pytest


@pytest.mark.tdd
def test_cached_and_uncached_generation_produce_same_logits():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_cached_and_uncached_generation_produce_same_logits`.

    Concepts à comprendre
    ---------------------
    - correctness KV
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    cache + inference

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 4.6.
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
def test_cached_and_uncached_greedy_generation_produce_same_tokens():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_cached_and_uncached_greedy_generation_produce_same_tokens`.

    Concepts à comprendre
    ---------------------
    - correctness KV
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    cache + inference

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 4.6.
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
def test_decode_position_advances_with_cache_length():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_decode_position_advances_with_cache_length`.

    Concepts à comprendre
    ---------------------
    - position cache
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    inference

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 4.7.
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


