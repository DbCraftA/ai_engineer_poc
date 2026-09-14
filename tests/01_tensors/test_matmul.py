import pytest


@pytest.mark.tdd
def test_matrix_multiplication_produces_expected_shape():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_matrix_multiplication_produces_expected_shape`.

    Concepts à comprendre
    ---------------------
    - matmul, M/N/K
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/calculators/flops.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 1.10.
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
def test_matmul_flops_can_be_estimated_from_mnk():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_matmul_flops_can_be_estimated_from_mnk`.

    Concepts à comprendre
    ---------------------
    - matmul, M/N/K
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/calculators/flops.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 1.10.
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
def test_gemv_is_matmul_with_single_output_row_or_vector_workload():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_gemv_is_matmul_with_single_output_row_or_vector_workload`.

    Concepts à comprendre
    ---------------------
    - GEMM / GEMV
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    documentation/calculators

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 1.11.
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


