import pytest


@pytest.mark.tdd
def test_hidden_dimension_is_split_across_attention_heads():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_hidden_dimension_is_split_across_attention_heads`.

    Concepts à comprendre
    ---------------------
    - MHA
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/attention/mha.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 2.7.
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
def test_attention_heads_are_concatenated_back_to_hidden_dimension():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_attention_heads_are_concatenated_back_to_hidden_dimension`.

    Concepts à comprendre
    ---------------------
    - MHA
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/attention/mha.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 2.7.
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


