import pytest


@pytest.mark.tdd
def test_contiguous_tensor_has_expected_strides():
    """
    Objectif
    --------
    Comprendre les strides standard d’un tenseur row-major contigu.

    Concepts à comprendre
    ---------------------
    - stride
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/layout.py

    Comportement à vérifier
    -----------------------
    Pour [B,T,C], les strides attendus sont [T*C, C, 1].

    Assertion attendue
    ------------------
    assert x.stride() == (T*C, C, 1)

    Hints d'implémentation
    ----------------------
    Utiliser x.stride() sur un tenseur contigu.

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
def test_stride_maps_indices_to_storage_offsets():
    """
    Objectif
    --------
    Relier index logique et offset mémoire.

    Concepts à comprendre
    ---------------------
    - stride
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/layout.py

    Comportement à vérifier
    -----------------------
    L’offset de x[b,t,c] doit suivre b*stride_B + t*stride_T + c*stride_C.

    Assertion attendue
    ------------------
    assert computed_offset == expected_offset

    Hints d'implémentation
    ----------------------
    Utiliser x.stride(), x.storage_offset() et une petite shape.

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
def test_transpose_changes_strides_without_reordering_storage():
    """
    Objectif
    --------
    Montrer que transpose change l’interprétation sans réordonner les données.

    Concepts à comprendre
    ---------------------
    - transpose
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/layout.py

    Comportement à vérifier
    -----------------------
    Le tensor transposé partage le storage et possède de nouveaux strides.

    Assertion attendue
    ------------------
    assert y.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()

    Hints d'implémentation
    ----------------------
    Utiliser torch.transpose ou x.transpose(dim0, dim1).

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
def test_contiguous_materializes_transposed_layout():
    """
    Objectif
    --------
    Comprendre quand une copie mémoire devient nécessaire.

    Concepts à comprendre
    ---------------------
    - contiguous, materialization
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/layout.py

    Comportement à vérifier
    -----------------------
    Après transpose, contiguous() doit produire un tensor contigu avec storage distinct.

    Assertion attendue
    ------------------
    assert y_contiguous.is_contiguous()

    Hints d'implémentation
    ----------------------
    Utiliser .transpose(...).contiguous().

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


