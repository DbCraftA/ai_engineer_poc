import pytest


@pytest.mark.tdd
def test_swiglu_uses_gate_and_up_projections():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `swiglu uses gate and up projections`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `swiglu uses gate and up projections`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `swiglu uses gate and up projections`.

    Hints d'implémentation
    ----------------------
    Commencer avec un exemple minimal, déterministe, sur CPU. Utiliser `torch.manual_seed(0)` si des valeurs aléatoires sont nécessaires. Le code cible indiqué par la roadmap est `src/inference_lab/nn/mlp/swiglu.py`.

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
    # `test_swiglu_uses_gate_and_up_projections`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "swiglu uses gate and up projections"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/nn/mlp/swiglu.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "swiglu uses gate and up projections"

    # Assert
    assert actual == expected


@pytest.mark.tdd
def test_swiglu_projects_back_to_hidden_dimension():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `swiglu projects back to hidden dimension`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `swiglu projects back to hidden dimension`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `swiglu projects back to hidden dimension`.

    Hints d'implémentation
    ----------------------
    Commencer avec un exemple minimal, déterministe, sur CPU. Utiliser `torch.manual_seed(0)` si des valeurs aléatoires sont nécessaires. Le code cible indiqué par la roadmap est `src/inference_lab/nn/mlp/swiglu.py`.

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
    # `test_swiglu_projects_back_to_hidden_dimension`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "swiglu projects back to hidden dimension"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/nn/mlp/swiglu.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "swiglu projects back to hidden dimension"

    # Assert
    assert actual == expected


