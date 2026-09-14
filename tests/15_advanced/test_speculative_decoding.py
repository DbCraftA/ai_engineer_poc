import pytest


@pytest.mark.tdd
def test_verified_speculative_tokens_match_target_model_distribution():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_verified_speculative_tokens_match_target_model_distribution`.

    Concepts à comprendre
    ---------------------
    - speculative decoding
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    future speculative decoding

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 15.4.
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
def test_acceptance_rate_is_computed_from_verified_draft_tokens():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_acceptance_rate_is_computed_from_verified_draft_tokens`.

    Concepts à comprendre
    ---------------------
    - draft acceptance
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    future speculative decoding

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 15.5.
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


