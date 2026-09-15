import pytest


@pytest.mark.tdd
def test_linear_layer_flops_are_estimated_from_matrix_dimensions():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `linear layer flops are estimated from matrix dimensions`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `linear layer flops are estimated from matrix dimensions`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `linear layer flops are estimated from matrix dimensions`.

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
    # `test_linear_layer_flops_are_estimated_from_matrix_dimensions`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "linear layer flops are estimated from matrix dimensions"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/calculators/flops.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "linear layer flops are estimated from matrix dimensions"

    # Assert
    assert actual == expected


@pytest.mark.tdd
def test_model_decode_flops_can_be_estimated_from_architecture():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `model decode flops can be estimated from architecture`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `model decode flops can be estimated from architecture`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `model decode flops can be estimated from architecture`.

    Hints d'implémentation
    ----------------------
    Commencer avec une formule fermée sur de petits entiers, par exemple `2 * M * N * K` pour un matmul dense. Le code cible indiqué par la roadmap est `calculator`.

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
    # `test_model_decode_flops_can_be_estimated_from_architecture`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "model decode flops can be estimated from architecture"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `calculator`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "model decode flops can be estimated from architecture"

    # Assert
    assert actual == expected


