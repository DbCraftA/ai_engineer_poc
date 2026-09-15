import pytest


@pytest.mark.tdd
@pytest.mark.model
def test_embedding_maps_token_ids_to_hidden_vectors():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `embedding maps token ids to hidden vectors`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `embedding maps token ids to hidden vectors`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `embedding maps token ids to hidden vectors`.

    Hints d'implémentation
    ----------------------
    Commencer avec un exemple minimal, déterministe, sur CPU. Utiliser `torch.manual_seed(0)` si des valeurs aléatoires sont nécessaires. Le code cible indiqué par la roadmap est `src/inference_lab/nn/embeddings.py`.

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
    # `test_embedding_maps_token_ids_to_hidden_vectors`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "embedding maps token ids to hidden vectors"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/nn/embeddings.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "embedding maps token ids to hidden vectors"

    # Assert
    assert actual == expected


