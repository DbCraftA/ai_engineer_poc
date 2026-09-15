import pytest


@pytest.mark.tdd
def test_tensor_shape_represents_logical_dimensions():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un tenseur PyTorch représentant des hidden states de Transformer, la propriété `shape` expose bien les dimensions logiques dans l'ordre `[batch_size, sequence_length, hidden_size]`.

    Pourquoi c'est important
    ------------------------
    Cette convention est centrale en inférence LLM : les embeddings, les blocs Transformer et le LM head manipulent presque toujours des tenseurs organisés autour de `[B, T, C]`. Savoir lire cette shape permet de suivre le calcul sans deviner.

    Comportement à vérifier
    -----------------------
    Étant donné un tenseur créé avec `torch.zeros(2, 3, 4)`, quand on lit `x.shape`, alors PyTorch doit retourner `torch.Size([2, 3, 4])`.

    Assertion attendue
    ------------------
    `assert x.shape == torch.Size([batch_size, sequence_length, hidden_size])`

    Hints d'implémentation
    ----------------------
    Utiliser `torch.zeros(batch_size, sequence_length, hidden_size)`, lire `x.shape`, puis comparer avec `torch.Size([...])`.

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
    import torch

    batch_size = 2
    sequence_length = 3
    hidden_size = 4

    # Act
    x = torch.zeros(batch_size, sequence_length, hidden_size)

    # Assert
    assert x.shape == torch.Size([batch_size, sequence_length, hidden_size])


@pytest.mark.tdd
def test_numel_is_product_of_dimensions():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `numel is product of dimensions`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `numel is product of dimensions`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `numel is product of dimensions`.

    Hints d'implémentation
    ----------------------
    Commencer avec un exemple minimal, déterministe, sur CPU. Utiliser `torch.manual_seed(0)` si des valeurs aléatoires sont nécessaires. Le code cible indiqué par la roadmap est `src/inference_lab/tensors/inspection.py`.

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
    # `test_numel_is_product_of_dimensions`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "numel is product of dimensions"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/tensors/inspection.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "numel is product of dimensions"

    # Assert
    assert actual == expected


@pytest.mark.tdd
def test_view_shares_storage_with_source_tensor():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `view shares storage with source tensor`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `view shares storage with source tensor`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `view shares storage with source tensor`.

    Hints d'implémentation
    ----------------------
    Commencer avec un exemple minimal, déterministe, sur CPU. Utiliser `torch.manual_seed(0)` si des valeurs aléatoires sont nécessaires. Le code cible indiqué par la roadmap est `src/inference_lab/tensors/inspection.py`.

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
    # `test_view_shares_storage_with_source_tensor`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "view shares storage with source tensor"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/tensors/inspection.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "view shares storage with source tensor"

    # Assert
    assert actual == expected


@pytest.mark.tdd
def test_clone_owns_independent_storage():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `clone owns independent storage`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `clone owns independent storage`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `clone owns independent storage`.

    Hints d'implémentation
    ----------------------
    Commencer avec un exemple minimal, déterministe, sur CPU. Utiliser `torch.manual_seed(0)` si des valeurs aléatoires sont nécessaires. Le code cible indiqué par la roadmap est `src/inference_lab/tensors/inspection.py`.

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
    # `test_clone_owns_independent_storage`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "clone owns independent storage"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/tensors/inspection.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "clone owns independent storage"

    # Assert
    assert actual == expected


