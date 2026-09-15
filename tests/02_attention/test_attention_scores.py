import pytest


@pytest.mark.tdd
def test_attention_scores_compare_every_query_with_every_key():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour des tenseurs `q` et `k`, le produit `Q @ Kᵀ` compare chaque position query avec chaque position key.

    Pourquoi c'est important
    ------------------------
    C'est le cœur de l'attention : chaque token courant produit une query qui est comparée aux keys des tokens visibles. La matrice `[query_length, key_length]` indique quelles positions peuvent influencer chaque query.

    Comportement à vérifier
    -----------------------
    Étant donné `q` de shape `[B, H, Tq, D]` et `k` de shape `[B, H, Tk, D]`, quand on calcule `q @ k.transpose(-2, -1)`, alors le résultat doit avoir la shape `[B, H, Tq, Tk]`.

    Assertion attendue
    ------------------
    `assert scores.shape == torch.Size([batch_size, num_heads, query_length, key_length])`

    Hints d'implémentation
    ----------------------
    Utiliser `torch.randn`, `k.transpose(-2, -1)` et l'opérateur `@`. Ne pas ajouter de scaling, de mask ou de softmax dans ce test : il vérifie uniquement la géométrie de `QKᵀ`.

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
    num_heads = 4
    query_length = 3
    key_length = 5
    head_dim = 8

    q = torch.randn(batch_size, num_heads, query_length, head_dim)
    k = torch.randn(batch_size, num_heads, key_length, head_dim)

    # Act
    scores = q @ k.transpose(-2, -1)

    # Assert
    assert scores.shape == torch.Size([batch_size, num_heads, query_length, key_length])


@pytest.mark.tdd
def test_attention_scores_are_scaled_by_inverse_sqrt_head_dimension():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `attention scores are scaled by inverse sqrt head dimension`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `attention scores are scaled by inverse sqrt head dimension`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `attention scores are scaled by inverse sqrt head dimension`.

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
    # `test_attention_scores_are_scaled_by_inverse_sqrt_head_dimension`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "attention scores are scaled by inverse sqrt head dimension"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/nn/attention/naive.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "attention scores are scaled by inverse sqrt head dimension"

    # Assert
    assert actual == expected


