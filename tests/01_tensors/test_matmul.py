import pytest


@pytest.mark.tdd
def test_matrix_multiplication_produces_expected_shape():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `matrix multiplication produces expected shape`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `matrix multiplication produces expected shape`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `matrix multiplication produces expected shape`.

    Hints d'implémentation
    ----------------------
    Utiliser de petits tenseurs CPU, lire `.shape`, puis comparer avec `torch.Size([...])`. Le code cible indiqué par la roadmap est `src/inference_lab/calculators/flops.py`.

    TDD
    ---
    1. supprimer `pytest.skip(...)` ;
    2. conserver ou affiner l'Arrange / Act / Assert ci-dessous ;
    3. obtenir RED si le code cible n'existe pas encore ou si le comportement est faux ;
    4. implémenter le minimum dans `src/` ;
    5. obtenir GREEN ;
    6. refactorer sans changer le comportement.
    """

    pytest.skip("Roadmap TDD — section pas encore activée")

    # Arrange
    # Construire ici un exemple minimal qui rend visible le comportement :
    # `test_matrix_multiplication_produces_expected_shape`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "matrix multiplication produces expected shape"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/calculators/flops.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "matrix multiplication produces expected shape"

    # Assert
    assert actual == expected


@pytest.mark.tdd
def test_matmul_flops_can_be_estimated_from_mnk():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `matmul flops can be estimated from mnk`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `matmul flops can be estimated from mnk`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `matmul flops can be estimated from mnk`.

    Hints d'implémentation
    ----------------------
    Commencer avec une formule fermée sur de petits entiers, par exemple `2 * M * N * K` pour un matmul dense. Le code cible indiqué par la roadmap est `src/inference_lab/calculators/flops.py`.

    TDD
    ---
    1. supprimer `pytest.skip(...)` ;
    2. conserver ou affiner l'Arrange / Act / Assert ci-dessous ;
    3. obtenir RED si le code cible n'existe pas encore ou si le comportement est faux ;
    4. implémenter le minimum dans `src/` ;
    5. obtenir GREEN ;
    6. refactorer sans changer le comportement.
    """

    pytest.skip("Roadmap TDD — section pas encore activée")

    # Arrange
    # Construire ici un exemple minimal qui rend visible le comportement :
    # `test_matmul_flops_can_be_estimated_from_mnk`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "matmul flops can be estimated from mnk"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/calculators/flops.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "matmul flops can be estimated from mnk"

    # Assert
    assert actual == expected


@pytest.mark.tdd
def test_gemv_is_matmul_with_single_output_row_or_vector_workload():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `gemv is matmul with single output row or vector workload`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `gemv is matmul with single output row or vector workload`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `gemv is matmul with single output row or vector workload`.

    Hints d'implémentation
    ----------------------
    Commencer avec un exemple minimal, déterministe, sur CPU. Utiliser `torch.manual_seed(0)` si des valeurs aléatoires sont nécessaires. Le code cible indiqué par la roadmap est `documentation/calculators`.

    TDD
    ---
    1. supprimer `pytest.skip(...)` ;
    2. conserver ou affiner l'Arrange / Act / Assert ci-dessous ;
    3. obtenir RED si le code cible n'existe pas encore ou si le comportement est faux ;
    4. implémenter le minimum dans `src/` ;
    5. obtenir GREEN ;
    6. refactorer sans changer le comportement.
    """

    pytest.skip("Roadmap TDD — section pas encore activée")

    # Arrange
    # Construire ici un exemple minimal qui rend visible le comportement :
    # `test_gemv_is_matmul_with_single_output_row_or_vector_workload`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "gemv is matmul with single output row or vector workload"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `documentation/calculators`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "gemv is matmul with single output row or vector workload"

    # Assert
    assert actual == expected


