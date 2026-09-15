import pytest


@pytest.mark.tdd
def test_long_prefill_can_be_split_into_multiple_chunks():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `long prefill can be split into multiple chunks`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `long prefill can be split into multiple chunks`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `long prefill can be split into multiple chunks`.

    Hints d'implémentation
    ----------------------
    Construire un prompt complet de longueur `T` ; vérifier que tous les tokens du prompt sont traités en une étape. Le code cible indiqué par la roadmap est `scheduler`.

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
    # `test_long_prefill_can_be_split_into_multiple_chunks`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "long prefill can be split into multiple chunks"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `scheduler`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "long prefill can be split into multiple chunks"

    # Assert
    assert actual == expected


