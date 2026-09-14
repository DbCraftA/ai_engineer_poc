import pytest


@pytest.mark.tdd
def test_contiguous_tensor_has_expected_strides():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_contiguous_tensor_has_expected_strides`.

    Concepts à comprendre
    ---------------------
    - stride
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/layout.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 1.4.
    Lorsque la section sera activée, elle sera remplacée par un vrai Arrange / Act / Assert.

    Critère de réussite
    -------------------
    Le test doit d'abord échouer en RED pour une raison pertinente, puis passer en GREEN
    après l'implémentation minimale dans src/.

    TDD
    ---
    1. supprimer pytest.skip()
    2. écrire l'assertion attendue
    3. obtenir RED
    4. implémenter le minimum dans src/
    5. obtenir GREEN
    6. refactorer sans changer le comportement
    """
    pytest.skip("Roadmap TDD — section pas encore activée")


@pytest.mark.tdd
def test_stride_maps_indices_to_storage_offsets():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_stride_maps_indices_to_storage_offsets`.

    Concepts à comprendre
    ---------------------
    - stride
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/layout.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 1.4.
    Lorsque la section sera activée, elle sera remplacée par un vrai Arrange / Act / Assert.

    Critère de réussite
    -------------------
    Le test doit d'abord échouer en RED pour une raison pertinente, puis passer en GREEN
    après l'implémentation minimale dans src/.

    TDD
    ---
    1. supprimer pytest.skip()
    2. écrire l'assertion attendue
    3. obtenir RED
    4. implémenter le minimum dans src/
    5. obtenir GREEN
    6. refactorer sans changer le comportement
    """
    pytest.skip("Roadmap TDD — section pas encore activée")


@pytest.mark.tdd
def test_transpose_changes_strides_without_reordering_storage():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_transpose_changes_strides_without_reordering_storage`.

    Concepts à comprendre
    ---------------------
    - transpose
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/layout.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 1.5.
    Lorsque la section sera activée, elle sera remplacée par un vrai Arrange / Act / Assert.

    Critère de réussite
    -------------------
    Le test doit d'abord échouer en RED pour une raison pertinente, puis passer en GREEN
    après l'implémentation minimale dans src/.

    TDD
    ---
    1. supprimer pytest.skip()
    2. écrire l'assertion attendue
    3. obtenir RED
    4. implémenter le minimum dans src/
    5. obtenir GREEN
    6. refactorer sans changer le comportement
    """
    pytest.skip("Roadmap TDD — section pas encore activée")


@pytest.mark.tdd
def test_contiguous_materializes_transposed_layout():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_contiguous_materializes_transposed_layout`.

    Concepts à comprendre
    ---------------------
    - contiguous, materialization
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/layout.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 1.6.
    Lorsque la section sera activée, elle sera remplacée par un vrai Arrange / Act / Assert.

    Critère de réussite
    -------------------
    Le test doit d'abord échouer en RED pour une raison pertinente, puis passer en GREEN
    après l'implémentation minimale dans src/.

    TDD
    ---
    1. supprimer pytest.skip()
    2. écrire l'assertion attendue
    3. obtenir RED
    4. implémenter le minimum dans src/
    5. obtenir GREEN
    6. refactorer sans changer le comportement
    """
    pytest.skip("Roadmap TDD — section pas encore activée")


