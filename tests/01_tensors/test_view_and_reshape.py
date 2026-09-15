import pytest


@pytest.mark.tdd
def test_view_does_not_copy_when_layout_allows_it():
    """
    Objectif
    --------
    Comprendre que view est une réinterprétation possible uniquement si le layout le permet.

    Concepts à comprendre
    ---------------------
    - view, reshape
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/layout.py

    Comportement à vérifier
    -----------------------
    view sur un tensor contigu doit partager le storage.

    Assertion attendue
    ------------------
    assert y.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()

    Hints d'implémentation
    ----------------------
    Utiliser x.view(new_shape).

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
def test_reshape_may_materialize_when_required():
    """
    Objectif
    --------
    Comprendre que reshape peut copier si view est impossible.

    Concepts à comprendre
    ---------------------
    - view, reshape
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/layout.py

    Comportement à vérifier
    -----------------------
    reshape après transpose peut retourner un tensor contigu indépendant.

    Assertion attendue
    ------------------
    assert reshaped.shape == expected_shape

    Hints d'implémentation
    ----------------------
    Comparer .reshape() avec .view(), inspecter .is_contiguous().

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


