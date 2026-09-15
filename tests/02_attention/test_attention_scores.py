import pytest


@pytest.mark.tdd
def test_attention_scores_compare_every_query_with_every_key():
    """
    Objectif
    --------
    Montrer que QKᵀ compare chaque query token à chaque key token.

    Concepts à comprendre
    ---------------------
    - score QKᵀ
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/attention/naive.py

    Comportement à vérifier
    -----------------------
    Les scores doivent avoir une dimension query_length x key_length.

    Assertion attendue
    ------------------
    assert scores.shape == (B, H, Tq, Tk)

    Hints d'implémentation
    ----------------------
    Utiliser q @ k.transpose(-2, -1).

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


@pytest.mark.tdd
def test_attention_scores_are_scaled_by_inverse_sqrt_head_dimension():
    """
    Objectif
    --------
    Vérifier le scaling qui stabilise la softmax attention.

    Concepts à comprendre
    ---------------------
    - scaling
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/attention/naive.py

    Comportement à vérifier
    -----------------------
    Les scores doivent être divisés par sqrt(head_dim).

    Assertion attendue
    ------------------
    torch.testing.assert_close(scores, raw_scores / math.sqrt(head_dim))

    Hints d'implémentation
    ----------------------
    Utiliser math.sqrt et torch.testing.assert_close.

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


