import pytest


@pytest.mark.tdd
def test_matrix_multiplication_produces_expected_shape():
    """
    Objectif
    --------
    Vérifier la règle de shape du matmul MxK @ KxN.

    Concepts à comprendre
    ---------------------
    - matmul, M/N/K
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/calculators/flops.py

    Comportement à vérifier
    -----------------------
    Le résultat doit avoir la shape [M, N].

    Assertion attendue
    ------------------
    assert out.shape == (M, N)

    Hints d'implémentation
    ----------------------
    Utiliser torch.matmul ou opérateur @.

    Critère de réussite
    -------------------
    Le test doit d'abord échouer en RED pour une raison pertinente, puis passer en GREEN
    après l'implémentation minimale dans src/.

    TDD
    ---
    1. supprimer pytest.skip()
    2. construire un Arrange / Act / Assert minimal
    3. écrire l'assertion attendue
    4. obtenir RED
    5. implémenter le minimum dans src/
    6. obtenir GREEN
    7. refactorer sans changer le comportement
    """
    pytest.skip("Roadmap TDD — section pas encore activée")


@pytest.mark.tdd
def test_matmul_flops_can_be_estimated_from_mnk():
    """
    Objectif
    --------
    Relier matmul et coût en FLOPs.

    Concepts à comprendre
    ---------------------
    - matmul, M/N/K
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/calculators/flops.py

    Comportement à vérifier
    -----------------------
    Le coût d’un matmul dense est environ 2*M*N*K opérations.

    Assertion attendue
    ------------------
    assert flops == 2 * M * N * K

    Hints d'implémentation
    ----------------------
    Créer un helper dans calculators/flops.py plus tard.

    Critère de réussite
    -------------------
    Le test doit d'abord échouer en RED pour une raison pertinente, puis passer en GREEN
    après l'implémentation minimale dans src/.

    TDD
    ---
    1. supprimer pytest.skip()
    2. construire un Arrange / Act / Assert minimal
    3. écrire l'assertion attendue
    4. obtenir RED
    5. implémenter le minimum dans src/
    6. obtenir GREEN
    7. refactorer sans changer le comportement
    """
    pytest.skip("Roadmap TDD — section pas encore activée")


@pytest.mark.tdd
def test_gemv_is_matmul_with_single_output_row_or_vector_workload():
    """
    Objectif
    --------
    Différencier GEMM et GEMV pour comprendre le decode LLM.

    Concepts à comprendre
    ---------------------
    - GEMM / GEMV
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    documentation/calculators

    Comportement à vérifier
    -----------------------
    Un GEMV correspond à une multiplication matrice-vecteur, souvent [M,K] @ [K].

    Assertion attendue
    ------------------
    assert out.shape == (M,)

    Hints d'implémentation
    ----------------------
    Utiliser torch.mv ou matmul avec une dimension singleton.

    Critère de réussite
    -------------------
    Le test doit d'abord échouer en RED pour une raison pertinente, puis passer en GREEN
    après l'implémentation minimale dans src/.

    TDD
    ---
    1. supprimer pytest.skip()
    2. construire un Arrange / Act / Assert minimal
    3. écrire l'assertion attendue
    4. obtenir RED
    5. implémenter le minimum dans src/
    6. obtenir GREEN
    7. refactorer sans changer le comportement
    """
    pytest.skip("Roadmap TDD — section pas encore activée")


