import pytest


@pytest.mark.tdd
@pytest.mark.gpu
@pytest.mark.cuda
def test_gpu_allocation_increases_allocated_memory():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `gpu allocation increases allocated memory`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `gpu allocation increases allocated memory`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `gpu allocation increases allocated memory`.

    Hints d'implémentation
    ----------------------
    Utiliser `numel()`, `element_size()` et une formule explicite en bytes avant de créer une abstraction dans `src/`. Le code cible indiqué par la roadmap est `src/inference_lab/profiling/memory.py`.

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
    # `test_gpu_allocation_increases_allocated_memory`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "gpu allocation increases allocated memory"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/profiling/memory.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "gpu allocation increases allocated memory"

    # Assert
    assert actual == expected


@pytest.mark.tdd
@pytest.mark.gpu
@pytest.mark.cuda
def test_peak_memory_can_be_recorded():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `peak memory can be recorded`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `peak memory can be recorded`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `peak memory can be recorded`.

    Hints d'implémentation
    ----------------------
    Utiliser `numel()`, `element_size()` et une formule explicite en bytes avant de créer une abstraction dans `src/`. Le code cible indiqué par la roadmap est `src/inference_lab/profiling/memory.py`.

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
    # `test_peak_memory_can_be_recorded`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "peak memory can be recorded"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/profiling/memory.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "peak memory can be recorded"

    # Assert
    assert actual == expected


