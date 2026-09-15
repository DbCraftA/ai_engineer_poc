import pytest


@pytest.mark.tdd
@pytest.mark.model
def test_custom_llm_can_select_naive_or_sdpa_attention():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `custom llm can select naive or sdpa attention`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `custom llm can select naive or sdpa attention`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `custom llm can select naive or sdpa attention`.

    Hints d'implémentation
    ----------------------
    Utiliser des tenseurs `[B, H, T, D]`, documenter chaque axe et vérifier les shapes intermédiaires. Le code cible indiqué par la roadmap est `custom model`.

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
    # `test_custom_llm_can_select_naive_or_sdpa_attention`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "custom llm can select naive or sdpa attention"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `custom model`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "custom llm can select naive or sdpa attention"

    # Assert
    assert actual == expected


@pytest.mark.tdd
@pytest.mark.model
def test_equivalent_attention_backends_produce_close_outputs():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `equivalent attention backends produce close outputs`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `equivalent attention backends produce close outputs`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `equivalent attention backends produce close outputs`.

    Hints d'implémentation
    ----------------------
    Utiliser des tenseurs `[B, H, T, D]`, documenter chaque axe et vérifier les shapes intermédiaires. Le code cible indiqué par la roadmap est `custom model`.

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
    # `test_equivalent_attention_backends_produce_close_outputs`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "equivalent attention backends produce close outputs"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `custom model`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "equivalent attention backends produce close outputs"

    # Assert
    assert actual == expected


