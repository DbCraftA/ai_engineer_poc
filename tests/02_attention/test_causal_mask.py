import pytest


@pytest.mark.tdd
def test_causal_mask_prevents_future_tokens_from_contributing():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour une séquence autoregressive, un token à la position `i` ne peut pas lire les tokens futurs `j > i`.

    Pourquoi c'est important
    ------------------------
    Un decoder-only LLM doit respecter la causalité. Sans masque causal, le modèle pourrait tricher pendant l'entraînement et utiliser des tokens futurs pour prédire le token courant.

    Comportement à vérifier
    -----------------------
    Étant donné une matrice de scores `[T, T]`, quand on applique un masque causal triangulaire supérieur, alors toutes les positions futures doivent recevoir `-inf` avant la softmax.

    Assertion attendue
    ------------------
    `assert torch.isneginf(masked_scores[future_mask]).all()`

    Hints d'implémentation
    ----------------------
    Utiliser `torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)` puis `scores.masked_fill(mask, float('-inf'))`.

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

    sequence_length = 4
    scores = torch.zeros(sequence_length, sequence_length)
    future_mask = torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool),
        diagonal=1,
    )

    # Act
    masked_scores = scores.masked_fill(future_mask, float("-inf"))

    # Assert
    assert torch.isneginf(masked_scores[future_mask]).all()


