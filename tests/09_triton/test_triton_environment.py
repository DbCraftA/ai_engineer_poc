import pytest


@pytest.mark.tdd
@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.triton
def test_triton_tests_skip_when_triton_or_cuda_is_unavailable():
    """
    Objectif
    --------
    Spécifier clairement la connaissance: triton tests skip when triton or cuda is unavailable.

    Concepts à comprendre
    ---------------------
    - Triton disponibilité
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    infrastructure

    Comportement à vérifier
    -----------------------
    Le comportement lié à Triton disponibilité doit être observable avec un exemple minimal et déterministe.

    Assertion attendue
    ------------------
    assert condition_attendue  # à remplacer par une assertion concrète lors de l’activation

    Hints d'implémentation
    ----------------------
    Commencer avec torch.manual_seed(0), de petits tenseurs CPU et torch.testing.assert_close si flottant. Code cible: infrastructure.

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


