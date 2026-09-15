import pytest


@pytest.mark.tdd
@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.triton
def test_triton_tests_skip_when_triton_or_cuda_is_unavailable():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `triton tests skip when triton or cuda is unavailable`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `triton tests skip when triton or cuda is unavailable`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `triton tests skip when triton or cuda is unavailable`.

    Hints d'implémentation
    ----------------------
    Marquer le test avec `@pytest.mark.gpu` et `@pytest.mark.cuda`, puis utiliser `torch.cuda.is_available()` via `tests/conftest.py`. Le code cible indiqué par la roadmap est `infrastructure`.

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
    # `test_triton_tests_skip_when_triton_or_cuda_is_unavailable`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "triton tests skip when triton or cuda is unavailable"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `infrastructure`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "triton tests skip when triton or cuda is unavailable"

    # Assert
    assert actual == expected


