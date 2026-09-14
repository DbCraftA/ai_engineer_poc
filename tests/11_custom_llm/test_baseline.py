import pytest


@pytest.mark.tdd
@pytest.mark.model
def test_custom_llm_baseline_matches_declared_configuration():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_custom_llm_baseline_matches_declared_configuration`.

    Concepts à comprendre
    ---------------------
    - custom model baseline
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/models/custom_llm/

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 11.1.
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


