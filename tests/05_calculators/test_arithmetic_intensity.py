import pytest


@pytest.mark.tdd
def test_arithmetic_intensity_is_flops_divided_by_bytes():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `arithmetic intensity is flops divided by bytes`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `arithmetic intensity is flops divided by bytes`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `arithmetic intensity is flops divided by bytes`.

    Hints d'implémentation
    ----------------------
    Commencer avec une formule fermée sur de petits entiers, par exemple `2 * M * N * K` pour un matmul dense. Le code cible indiqué par la roadmap est `src/inference_lab/calculators/roofline.py`.

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
    # `test_arithmetic_intensity_is_flops_divided_by_bytes`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "arithmetic intensity is flops divided by bytes"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/calculators/roofline.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "arithmetic intensity is flops divided by bytes"

    # Assert
    assert actual == expected


@pytest.mark.tdd
def test_roofline_limit_is_minimum_of_compute_and_bandwidth_limits():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `roofline limit is minimum of compute and bandwidth limits`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `roofline limit is minimum of compute and bandwidth limits`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `roofline limit is minimum of compute and bandwidth limits`.

    Hints d'implémentation
    ----------------------
    Commencer avec un exemple minimal, déterministe, sur CPU. Utiliser `torch.manual_seed(0)` si des valeurs aléatoires sont nécessaires. Le code cible indiqué par la roadmap est `src/inference_lab/calculators/roofline.py`.

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
    # `test_roofline_limit_is_minimum_of_compute_and_bandwidth_limits`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "roofline limit is minimum of compute and bandwidth limits"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/calculators/roofline.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "roofline limit is minimum of compute and bandwidth limits"

    # Assert
    assert actual == expected


@pytest.mark.tdd
def test_low_arithmetic_intensity_is_classified_as_bandwidth_limited():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `low arithmetic intensity is classified as bandwidth limited`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `low arithmetic intensity is classified as bandwidth limited`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `low arithmetic intensity is classified as bandwidth limited`.

    Hints d'implémentation
    ----------------------
    Commencer avec un exemple minimal, déterministe, sur CPU. Utiliser `torch.manual_seed(0)` si des valeurs aléatoires sont nécessaires. Le code cible indiqué par la roadmap est `calculator`.

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
    # `test_low_arithmetic_intensity_is_classified_as_bandwidth_limited`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "low arithmetic intensity is classified as bandwidth limited"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `calculator`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "low arithmetic intensity is classified as bandwidth limited"

    # Assert
    assert actual == expected


