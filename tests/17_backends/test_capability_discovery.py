import pytest


@pytest.mark.tdd
def test_backend_reports_supported_dtypes_and_operations():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `backend reports supported dtypes and operations`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `backend reports supported dtypes and operations`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `backend reports supported dtypes and operations`.

    Hints d'implémentation
    ----------------------
    Utiliser `torch.empty(..., dtype=...)`, `.dtype`, `.element_size()` et `torch.testing.assert_close` si une comparaison numérique est nécessaire. Le code cible indiqué par la roadmap est `future backends`.

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
    # `test_backend_reports_supported_dtypes_and_operations`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "backend reports supported dtypes and operations"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `future backends`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "backend reports supported dtypes and operations"

    # Assert
    assert actual == expected


