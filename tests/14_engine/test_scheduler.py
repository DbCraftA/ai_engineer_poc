import pytest


@pytest.mark.tdd
def test_finished_sequences_can_leave_batch_while_others_continue():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `finished sequences can leave batch while others continue`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `finished sequences can leave batch while others continue`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `finished sequences can leave batch while others continue`.

    Hints d'implémentation
    ----------------------
    Commencer avec un exemple minimal, déterministe, sur CPU. Utiliser `torch.manual_seed(0)` si des valeurs aléatoires sont nécessaires. Le code cible indiqué par la roadmap est `scheduler`.

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
    # `test_finished_sequences_can_leave_batch_while_others_continue`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "finished sequences can leave batch while others continue"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `scheduler`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "finished sequences can leave batch while others continue"

    # Assert
    assert actual == expected


@pytest.mark.tdd
def test_scheduler_rejects_or_delays_requests_when_capacity_is_exhausted():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `scheduler rejects or delays requests when capacity is exhausted`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `scheduler rejects or delays requests when capacity is exhausted`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `scheduler rejects or delays requests when capacity is exhausted`.

    Hints d'implémentation
    ----------------------
    Commencer avec un exemple minimal, déterministe, sur CPU. Utiliser `torch.manual_seed(0)` si des valeurs aléatoires sont nécessaires. Le code cible indiqué par la roadmap est `scheduler`.

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
    # `test_scheduler_rejects_or_delays_requests_when_capacity_is_exhausted`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "scheduler rejects or delays requests when capacity is exhausted"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `scheduler`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "scheduler rejects or delays requests when capacity is exhausted"

    # Assert
    assert actual == expected


