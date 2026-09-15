import pytest


@pytest.mark.tdd
@pytest.mark.perf
def test_decode_benchmark_reports_time_per_output_token():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `decode benchmark reports time per output token`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `decode benchmark reports time per output token`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `decode benchmark reports time per output token`.

    Hints d'implémentation
    ----------------------
    Construire un exemple avec un prompt déjà prérempli puis un seul nouveau token ; vérifier que seule une position est ajoutée. Le code cible indiqué par la roadmap est `benchmark`.

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
    # `test_decode_benchmark_reports_time_per_output_token`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "decode benchmark reports time per output token"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `benchmark`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "decode benchmark reports time per output token"

    # Assert
    assert actual == expected


