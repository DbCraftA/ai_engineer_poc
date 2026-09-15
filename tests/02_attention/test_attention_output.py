import pytest


@pytest.mark.tdd
def test_attention_output_is_weighted_sum_of_values():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `attention output is weighted sum of values`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `attention output is weighted sum of values`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `attention output is weighted sum of values`.

    Hints d'implémentation
    ----------------------
    Utiliser des tenseurs `[B, H, T, D]`, documenter chaque axe et vérifier les shapes intermédiaires. Le code cible indiqué par la roadmap est `src/inference_lab/nn/attention/naive.py`.

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
    # `test_attention_output_is_weighted_sum_of_values`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "attention output is weighted sum of values"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/nn/attention/naive.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "attention output is weighted sum of values"

    # Assert
    assert actual == expected


