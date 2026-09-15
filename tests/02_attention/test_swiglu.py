import pytest


@pytest.mark.tdd
def test_swiglu_uses_gate_and_up_projections():
    """
    Objectif
    --------
    Comprendre le gating SwiGLU.

    Concepts à comprendre
    ---------------------
    - SwiGLU
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/mlp/swiglu.py

    Comportement à vérifier
    -----------------------
    La sortie intermédiaire doit être silu(gate) * up.

    Assertion attendue
    ------------------
    torch.testing.assert_close(hidden, torch.nn.functional.silu(gate) * up)

    Hints d'implémentation
    ----------------------
    Utiliser torch.nn.functional.silu.

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
def test_swiglu_projects_back_to_hidden_dimension():
    """
    Objectif
    --------
    Vérifier que le MLP revient à hidden_size après expansion.

    Concepts à comprendre
    ---------------------
    - SwiGLU
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/mlp/swiglu.py

    Comportement à vérifier
    -----------------------
    Le down projection doit produire [B,T,hidden_size].

    Assertion attendue
    ------------------
    assert out.shape == (B, T, hidden_size)

    Hints d'implémentation
    ----------------------
    Utiliser trois Linear: gate_proj, up_proj, down_proj.

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


