import pytest


@pytest.mark.tdd
@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_backend_matches_cpu_reference():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `cuda backend matches cpu reference`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `cuda backend matches cpu reference`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `cuda backend matches cpu reference`.

    Hints d'implémentation
    ----------------------
    Marquer le test avec `@pytest.mark.gpu` et `@pytest.mark.cuda`, puis utiliser `torch.cuda.is_available()` via `tests/conftest.py`. Le code cible indiqué par la roadmap est `future backends`.

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
    # `test_cuda_backend_matches_cpu_reference`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "cuda backend matches cpu reference"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `future backends`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "cuda backend matches cpu reference"

    # Assert
    assert actual == expected


