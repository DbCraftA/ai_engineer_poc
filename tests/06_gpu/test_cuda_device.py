import pytest


@pytest.mark.tdd
@pytest.mark.gpu
@pytest.mark.cuda
def test_tensor_can_be_created_on_cuda():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `tensor can be created on cuda`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `tensor can be created on cuda`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `tensor can be created on cuda`.

    Hints d'implémentation
    ----------------------
    Marquer le test avec `@pytest.mark.gpu` et `@pytest.mark.cuda`, puis utiliser `torch.cuda.is_available()` via `tests/conftest.py`. Le code cible indiqué par la roadmap est `helpers GPU`.

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
    # `test_tensor_can_be_created_on_cuda`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "tensor can be created on cuda"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `helpers GPU`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "tensor can be created on cuda"

    # Assert
    assert actual == expected


@pytest.mark.tdd
@pytest.mark.gpu
@pytest.mark.cuda
def test_cpu_and_cuda_tensors_report_different_devices():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `cpu and cuda tensors report different devices`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `cpu and cuda tensors report different devices`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `cpu and cuda tensors report different devices`.

    Hints d'implémentation
    ----------------------
    Marquer le test avec `@pytest.mark.gpu` et `@pytest.mark.cuda`, puis utiliser `torch.cuda.is_available()` via `tests/conftest.py`. Le code cible indiqué par la roadmap est `helpers GPU`.

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
    # `test_cpu_and_cuda_tensors_report_different_devices`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "cpu and cuda tensors report different devices"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `helpers GPU`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "cpu and cuda tensors report different devices"

    # Assert
    assert actual == expected


