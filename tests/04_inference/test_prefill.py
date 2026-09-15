import pytest


@pytest.mark.tdd
def test_prefill_processes_complete_prompt():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `prefill processes complete prompt`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `prefill processes complete prompt`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `prefill processes complete prompt`.

    Hints d'implémentation
    ----------------------
    Construire un prompt complet de longueur `T` ; vérifier que tous les tokens du prompt sont traités en une étape. Le code cible indiqué par la roadmap est `src/inference_lab/inference/prefill.py`.

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
    # `test_prefill_processes_complete_prompt`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "prefill processes complete prompt"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/inference/prefill.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "prefill processes complete prompt"

    # Assert
    assert actual == expected


@pytest.mark.tdd
def test_prefill_populates_initial_kv_cache():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `prefill populates initial kv cache`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `prefill populates initial kv cache`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `prefill populates initial kv cache`.

    Hints d'implémentation
    ----------------------
    Utiliser de petits tenseurs `[num_layers, batch, heads, seq, head_dim]` et vérifier explicitement la longueur de séquence stockée. Le code cible indiqué par la roadmap est `src/inference_lab/inference/prefill.py`.

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
    # `test_prefill_populates_initial_kv_cache`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "prefill populates initial kv cache"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/inference/prefill.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "prefill populates initial kv cache"

    # Assert
    assert actual == expected


