import pytest


@pytest.mark.tdd
def test_gqa_uses_fewer_kv_heads_than_query_heads():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_gqa_uses_fewer_kv_heads_than_query_heads`.

    Concepts à comprendre
    ---------------------
    - GQA
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/attention/gqa.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 2.8.
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
def test_multiple_query_heads_share_key_value_heads():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_multiple_query_heads_share_key_value_heads`.

    Concepts à comprendre
    ---------------------
    - GQA
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/attention/gqa.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 2.8.
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
def test_mqa_uses_single_key_value_head():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_mqa_uses_single_key_value_head`.

    Concepts à comprendre
    ---------------------
    - MQA
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/attention/gqa.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 2.9.
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


