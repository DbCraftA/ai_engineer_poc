import pytest


@pytest.mark.tdd
def test_naive_attention_matches_pytorch_sdpa_for_supported_case():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `naive attention matches pytorch sdpa for supported case`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `naive attention matches pytorch sdpa for supported case`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `naive attention matches pytorch sdpa for supported case`.

    Hints d'implémentation
    ----------------------
    Utiliser des tenseurs `[B, H, T, D]`, documenter chaque axe et vérifier les shapes intermédiaires. Le code cible indiqué par la roadmap est `attention`.

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
    # `test_naive_attention_matches_pytorch_sdpa_for_supported_case`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "naive attention matches pytorch sdpa for supported case"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `attention`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "naive attention matches pytorch sdpa for supported case"

    # Assert
    assert actual == expected


@pytest.mark.tdd
def test_causal_sdpa_matches_naive_causal_attention():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `causal sdpa matches naive causal attention`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `causal sdpa matches naive causal attention`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `causal sdpa matches naive causal attention`.

    Hints d'implémentation
    ----------------------
    Utiliser des tenseurs `[B, H, T, D]`, documenter chaque axe et vérifier les shapes intermédiaires. Le code cible indiqué par la roadmap est `attention`.

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
    # `test_causal_sdpa_matches_naive_causal_attention`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "causal sdpa matches naive causal attention"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `attention`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "causal sdpa matches naive causal attention"

    # Assert
    assert actual == expected


