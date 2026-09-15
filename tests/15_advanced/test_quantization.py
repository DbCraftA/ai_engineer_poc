import pytest


@pytest.mark.tdd
def test_quantized_weights_require_less_storage_than_fp16_weights():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `quantized weights require less storage than fp16 weights`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `quantized weights require less storage than fp16 weights`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `quantized weights require less storage than fp16 weights`.

    Hints d'implémentation
    ----------------------
    Commencer avec un exemple minimal, déterministe, sur CPU. Utiliser `torch.manual_seed(0)` si des valeurs aléatoires sont nécessaires. Le code cible indiqué par la roadmap est `future quantization`.

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
    # `test_quantized_weights_require_less_storage_than_fp16_weights`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "quantized weights require less storage than fp16 weights"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `future quantization`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "quantized weights require less storage than fp16 weights"

    # Assert
    assert actual == expected


@pytest.mark.tdd
def test_dequantized_output_remains_close_to_reference():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `dequantized output remains close to reference`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `dequantized output remains close to reference`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `dequantized output remains close to reference`.

    Hints d'implémentation
    ----------------------
    Commencer avec un exemple minimal, déterministe, sur CPU. Utiliser `torch.manual_seed(0)` si des valeurs aléatoires sont nécessaires. Le code cible indiqué par la roadmap est `future quantization`.

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
    # `test_dequantized_output_remains_close_to_reference`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "dequantized output remains close to reference"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `future quantization`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "dequantized output remains close to reference"

    # Assert
    assert actual == expected


