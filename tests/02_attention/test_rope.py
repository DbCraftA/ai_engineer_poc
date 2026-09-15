import pytest


@pytest.mark.tdd
def test_rope_preserves_vector_norm():
    """
    Objectif
    --------
    Vérifier que RoPE est une rotation et ne change pas la norme.

    Concepts à comprendre
    ---------------------
    - RoPE
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/positional/rope.py

    Comportement à vérifier
    -----------------------
    La norme avant/après RoPE doit rester identique.

    Assertion attendue
    ------------------
    torch.testing.assert_close(x.norm(dim=-1), y.norm(dim=-1))

    Hints d'implémentation
    ----------------------
    Utiliser sin/cos, split pair/impair ou représentation complexe.

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
def test_rope_changes_representation_according_to_position():
    """
    Objectif
    --------
    Montrer que RoPE encode la position dans Q/K.

    Concepts à comprendre
    ---------------------
    - RoPE
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/positional/rope.py

    Comportement à vérifier
    -----------------------
    Deux positions différentes doivent produire des représentations différentes.

    Assertion attendue
    ------------------
    assert not torch.allclose(y_pos0, y_pos1)

    Hints d'implémentation
    ----------------------
    Utiliser position_ids et fréquences RoPE.

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


