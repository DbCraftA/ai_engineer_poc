import pytest


@pytest.mark.tdd
def test_gqa_kv_cache_stores_only_kv_heads_not_query_heads():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `gqa kv cache stores only kv heads not query heads`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `gqa kv cache stores only kv heads not query heads`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `gqa kv cache stores only kv heads not query heads`.

    Hints d'implémentation
    ----------------------
    Utiliser de petits tenseurs `[num_layers, batch, heads, seq, head_dim]` et vérifier explicitement la longueur de séquence stockée. Le code cible indiqué par la roadmap est `cache`.

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
    # `test_gqa_kv_cache_stores_only_kv_heads_not_query_heads`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "gqa kv cache stores only kv heads not query heads"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `cache`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "gqa kv cache stores only kv heads not query heads"

    # Assert
    assert actual == expected


