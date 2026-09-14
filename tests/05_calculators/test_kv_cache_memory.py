import pytest


@pytest.mark.tdd
def test_kv_cache_memory_matches_layers_heads_tokens_formula():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_kv_cache_memory_matches_layers_heads_tokens_formula`.

    Concepts à comprendre
    ---------------------
    - KV cache memory
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/calculators/kv_cache_memory.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 5.2.
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
def test_kv_cache_memory_scales_linearly_with_context_length():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_kv_cache_memory_scales_linearly_with_context_length`.

    Concepts à comprendre
    ---------------------
    - impact contexte
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    calculator

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 5.3.
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
def test_reducing_kv_heads_reduces_kv_cache_memory_proportionally():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_reducing_kv_heads_reduces_kv_cache_memory_proportionally`.

    Concepts à comprendre
    ---------------------
    - impact GQA
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    calculator

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 5.4.
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


