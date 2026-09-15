import pytest


@pytest.mark.tdd
@pytest.mark.model
def test_qwen_weights_can_be_loaded_into_internal_modules():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `qwen weights can be loaded into internal modules`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `qwen weights can be loaded into internal modules`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `qwen weights can be loaded into internal modules`.

    Hints d'implémentation
    ----------------------
    Commencer avec un exemple minimal, déterministe, sur CPU. Utiliser `torch.manual_seed(0)` si des valeurs aléatoires sont nécessaires. Le code cible indiqué par la roadmap est `src/inference_lab/models/qwen/weight_loader.py`.

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
    # `test_qwen_weights_can_be_loaded_into_internal_modules`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "qwen weights can be loaded into internal modules"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/models/qwen/weight_loader.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "qwen weights can be loaded into internal modules"

    # Assert
    assert actual == expected


