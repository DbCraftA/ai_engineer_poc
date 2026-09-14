import pytest


@pytest.mark.tdd
def test_dtype_controls_bytes_per_element():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_dtype_controls_bytes_per_element`.

    Concepts à comprendre
    ---------------------
    - FP32, FP16, BF16
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/dtypes.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 1.8.
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
def test_reduced_precision_changes_numerical_accuracy():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_reduced_precision_changes_numerical_accuracy`.

    Concepts à comprendre
    ---------------------
    - FP32, FP16, BF16
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/dtypes.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 1.8.
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
def test_tensor_memory_equals_numel_times_element_size():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_tensor_memory_equals_numel_times_element_size`.

    Concepts à comprendre
    ---------------------
    - mémoire tensor
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/memory.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 1.9.
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


