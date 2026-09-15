import pytest


@pytest.mark.tdd
def test_hidden_dimension_is_split_across_attention_heads():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `hidden dimension is split across attention heads`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `hidden dimension is split across attention heads`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `hidden dimension is split across attention heads`.

    Hints d'implémentation
    ----------------------
    Utiliser des tenseurs `[B, H, T, D]`, documenter chaque axe et vérifier les shapes intermédiaires. Le code cible indiqué par la roadmap est `src/inference_lab/nn/attention/mha.py`.

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
    # `test_hidden_dimension_is_split_across_attention_heads`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "hidden dimension is split across attention heads"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/nn/attention/mha.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "hidden dimension is split across attention heads"

    # Assert
    assert actual == expected


@pytest.mark.tdd
def test_attention_heads_are_concatenated_back_to_hidden_dimension():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `attention heads are concatenated back to hidden dimension`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `attention heads are concatenated back to hidden dimension`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `attention heads are concatenated back to hidden dimension`.

    Hints d'implémentation
    ----------------------
    Utiliser des tenseurs `[B, H, T, D]`, documenter chaque axe et vérifier les shapes intermédiaires. Le code cible indiqué par la roadmap est `src/inference_lab/nn/attention/mha.py`.

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
    # `test_attention_heads_are_concatenated_back_to_hidden_dimension`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "attention heads are concatenated back to hidden dimension"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/nn/attention/mha.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "attention heads are concatenated back to hidden dimension"

    # Assert
    assert actual == expected


