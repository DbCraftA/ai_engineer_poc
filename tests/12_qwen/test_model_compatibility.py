import pytest


@pytest.mark.tdd
@pytest.mark.model
def test_internal_qwen_model_matches_reference_logits_on_small_input():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `internal qwen model matches reference logits on small input`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `internal qwen model matches reference logits on small input`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `internal qwen model matches reference logits on small input`.

    Hints d'implémentation
    ----------------------
    Vérifier que les logits ont la shape `[batch_size, sequence_length, vocab_size]` et représentent un score par token du vocabulaire. Le code cible indiqué par la roadmap est `modèle Qwen`.

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
    # `test_internal_qwen_model_matches_reference_logits_on_small_input`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "internal qwen model matches reference logits on small input"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `modèle Qwen`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "internal qwen model matches reference logits on small input"

    # Assert
    assert actual == expected


