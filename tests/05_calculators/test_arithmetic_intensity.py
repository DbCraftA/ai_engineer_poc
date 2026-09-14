import pytest


@pytest.mark.tdd
def test_arithmetic_intensity_is_flops_divided_by_bytes():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_arithmetic_intensity_is_flops_divided_by_bytes`.

    Concepts à comprendre
    ---------------------
    - arithmetic intensity
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/calculators/roofline.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 5.8.
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
def test_roofline_limit_is_minimum_of_compute_and_bandwidth_limits():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_roofline_limit_is_minimum_of_compute_and_bandwidth_limits`.

    Concepts à comprendre
    ---------------------
    - roofline
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/calculators/roofline.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 5.9.
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
def test_low_arithmetic_intensity_is_classified_as_bandwidth_limited():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_low_arithmetic_intensity_is_classified_as_bandwidth_limited`.

    Concepts à comprendre
    ---------------------
    - bottleneck
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    calculator

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 5.10.
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


