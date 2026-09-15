import pytest


@pytest.mark.tdd
@pytest.mark.distributed
def test_all_gather_reconstructs_all_shards():
    """
    Objectif
    --------
    Spécifier clairement la connaissance: all gather reconstructs all shards.

    Concepts à comprendre
    ---------------------
    - AllGather
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    future distributed

    Comportement à vérifier
    -----------------------
    Le comportement lié à AllGather doit être observable avec un exemple minimal et déterministe.

    Assertion attendue
    ------------------
    assert condition_attendue  # à remplacer par une assertion concrète lors de l’activation

    Hints d'implémentation
    ----------------------
    Commencer avec torch.manual_seed(0), de petits tenseurs CPU et torch.testing.assert_close si flottant. Code cible: future distributed.

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
@pytest.mark.distributed
def test_all_reduce_matches_single_device_reference_sum():
    """
    Objectif
    --------
    Spécifier clairement la connaissance: all reduce matches single device reference sum.

    Concepts à comprendre
    ---------------------
    - AllReduce
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    future distributed

    Comportement à vérifier
    -----------------------
    Le comportement lié à AllReduce doit être observable avec un exemple minimal et déterministe.

    Assertion attendue
    ------------------
    assert condition_attendue  # à remplacer par une assertion concrète lors de l’activation

    Hints d'implémentation
    ----------------------
    Commencer avec torch.manual_seed(0), de petits tenseurs CPU et torch.testing.assert_close si flottant. Code cible: future distributed.

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


