import pytest


@pytest.mark.tdd
def test_qkv_projections_produce_expected_shapes():
    """
    Objectif
    --------
    Vérifier que les projections Q/K/V transforment hidden_states en tenseurs d’attention.

    Concepts à comprendre
    ---------------------
    - projection linéaire Q/K/V
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/attention/qkv.py

    Comportement à vérifier
    -----------------------
    q, k, v doivent avoir les shapes attendues selon heads et head_dim.

    Assertion attendue
    ------------------
    assert q.shape == (B, num_heads, T, head_dim)

    Hints d'implémentation
    ----------------------
    Utiliser torch.nn.Linear et reshape/transpose.

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


