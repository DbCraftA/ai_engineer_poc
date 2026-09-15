import pytest


@pytest.mark.tdd
@pytest.mark.model
def test_custom_llm_can_switch_between_mha_and_gqa():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `custom llm can switch between mha and gqa`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `custom llm can switch between mha and gqa`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `custom llm can switch between mha and gqa`.

    Hints d'implémentation
    ----------------------
    Commencer avec un exemple minimal, déterministe, sur CPU. Utiliser `torch.manual_seed(0)` si des valeurs aléatoires sont nécessaires. Le code cible indiqué par la roadmap est `custom model`.

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
    # `test_custom_llm_can_switch_between_mha_and_gqa`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "custom llm can switch between mha and gqa"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `custom model`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "custom llm can switch between mha and gqa"

    # Assert
    assert actual == expected


