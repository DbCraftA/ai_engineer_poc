import pytest


@pytest.mark.tdd
@pytest.mark.distributed
def test_attention_heads_can_be_partitioned_across_devices():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `attention heads can be partitioned across devices`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `attention heads can be partitioned across devices`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `attention heads can be partitioned across devices`.

    Hints d'implémentation
    ----------------------
    Utiliser des tenseurs `[B, H, T, D]`, documenter chaque axe et vérifier les shapes intermédiaires. Le code cible indiqué par la roadmap est `future distributed`.

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
    # `test_attention_heads_can_be_partitioned_across_devices`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "attention heads can be partitioned across devices"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `future distributed`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "attention heads can be partitioned across devices"

    # Assert
    assert actual == expected


