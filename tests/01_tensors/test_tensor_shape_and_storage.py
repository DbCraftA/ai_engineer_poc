import pytest


@pytest.mark.tdd
def test_tensor_shape_represents_logical_dimensions():
    """
    Objectif
    --------
    Montrer que la shape décrit les axes logiques d’un tenseur.

    Concepts à comprendre
    ---------------------
    - tensor, shape, dimension, numel
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/inspection.py

    Comportement à vérifier
    -----------------------
    Un tenseur [B, T, C] expose batch, sequence et hidden dimension.

    Assertion attendue
    ------------------
    assert x.shape == torch.Size([B, T, C])

    Hints d'implémentation
    ----------------------
    Utiliser torch.zeros ou torch.arange puis .shape.

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
def test_numel_is_product_of_dimensions():
    """
    Objectif
    --------
    Relier la shape au nombre total d’éléments.

    Concepts à comprendre
    ---------------------
    - tensor, shape, dimension, numel
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/inspection.py

    Comportement à vérifier
    -----------------------
    numel doit être le produit des dimensions logiques.

    Assertion attendue
    ------------------
    assert x.numel() == B * T * C

    Hints d'implémentation
    ----------------------
    Utiliser torch.Tensor.numel().

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
def test_view_shares_storage_with_source_tensor():
    """
    Objectif
    --------
    Comprendre qu’une view peut partager la même mémoire que sa source.

    Concepts à comprendre
    ---------------------
    - storage, partage mémoire
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/inspection.py

    Comportement à vérifier
    -----------------------
    Après view, les pointeurs de storage doivent être identiques.

    Assertion attendue
    ------------------
    assert view.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()

    Hints d'implémentation
    ----------------------
    Préférer untyped_storage().data_ptr() à storage().data_ptr().

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
def test_clone_owns_independent_storage():
    """
    Objectif
    --------
    Distinguer view et copie réelle.

    Concepts à comprendre
    ---------------------
    - storage, partage mémoire
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/inspection.py

    Comportement à vérifier
    -----------------------
    clone doit produire un storage indépendant tout en gardant les mêmes valeurs.

    Assertion attendue
    ------------------
    assert clone.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()

    Hints d'implémentation
    ----------------------
    Utiliser x.clone() et torch.testing.assert_close.

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


