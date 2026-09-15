import pytest


@pytest.mark.tdd
def test_causal_mask_prevents_future_tokens_from_contributing():
    """
    Objectif
    --------
    Garantir qu’un token ne peut pas voir les positions futures.

    Concepts à comprendre
    ---------------------
    - causal mask
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/attention/mask.py

    Comportement à vérifier
    -----------------------
    Les positions futures doivent être masquées avant softmax.

    Assertion attendue
    ------------------
    assert masked_scores[..., future_positions] == -inf ou probs == 0

    Hints d'implémentation
    ----------------------
    Utiliser torch.triu, masked_fill, float("-inf").

    Critère de réussite
    -------------------
    Le test doit d'abord échouer en RED pour une raison pertinente, puis passer en GREEN
    après l'implémentation minimale dans src/.

    TDD
    ---
    1. supprimer pytest.skip()
    2. construire un Arrange / Act / Assert minimal
    3. écrire l'assertion attendue
    4. obtenir RED
    5. implémenter le minimum dans src/
    6. obtenir GREEN
    7. refactorer sans changer le comportement
    """
    pytest.skip("Roadmap TDD — section pas encore activée")


