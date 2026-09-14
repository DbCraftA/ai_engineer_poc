import pytest


@pytest.mark.tdd
def test_kv_cache_stores_keys_and_values_for_every_layer():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_kv_cache_stores_keys_and_values_for_every_layer`.

    Concepts à comprendre
    ---------------------
    - structure KV cache
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/cache/kv_cache.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 4.2.
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
def test_kv_cache_grows_one_position_per_decode_step():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_kv_cache_grows_one_position_per_decode_step`.

    Concepts à comprendre
    ---------------------
    - croissance cache
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/cache/kv_cache.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 4.3.
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


