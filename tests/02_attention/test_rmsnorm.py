import pytest


@pytest.mark.tdd
def test_rmsnorm_preserves_shape():
    """
    Objectif
    --------
    Vérifier qu’une normalisation ne change pas la shape.

    Concepts à comprendre
    ---------------------
    - RMSNorm
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/normalization/rmsnorm.py

    Comportement à vérifier
    -----------------------
    RMSNorm(x) doit retourner la même shape que x.

    Assertion attendue
    ------------------
    assert y.shape == x.shape

    Hints d'implémentation
    ----------------------
    Formule: x * rsqrt(mean(x*x)+eps) * weight.

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
def test_rmsnorm_matches_reference_formula():
    """
    Objectif
    --------
    Valider RMSNorm contre sa formule mathématique explicite.

    Concepts à comprendre
    ---------------------
    - RMSNorm
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/normalization/rmsnorm.py

    Comportement à vérifier
    -----------------------
    La sortie doit matcher la formule PyTorch naïve.

    Assertion attendue
    ------------------
    torch.testing.assert_close(y, expected)

    Hints d'implémentation
    ----------------------
    Utiliser torch.rsqrt, mean(dim=-1, keepdim=True).

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


