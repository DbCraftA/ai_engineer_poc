import pytest


@pytest.mark.tdd
def test_dtype_controls_bytes_per_element():
    """
    Objectif
    --------
    Relier dtype et coût mémoire élémentaire.

    Concepts à comprendre
    ---------------------
    - FP32, FP16, BF16
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/dtypes.py

    Comportement à vérifier
    -----------------------
    float32, float16 et bfloat16 doivent exposer des element_size différents.

    Assertion attendue
    ------------------
    assert torch.float32 tensor.element_size() == 4

    Hints d'implémentation
    ----------------------
    Utiliser torch.empty(..., dtype=...).element_size().

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
def test_reduced_precision_changes_numerical_accuracy():
    """
    Objectif
    --------
    Observer qu’un dtype réduit modifie la précision numérique.

    Concepts à comprendre
    ---------------------
    - FP32, FP16, BF16
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/dtypes.py

    Comportement à vérifier
    -----------------------
    Une opération en float16/bfloat16 peut différer légèrement de float32.

    Assertion attendue
    ------------------
    torch.testing.assert_close(..., atol=..., rtol=...)

    Hints d'implémentation
    ----------------------
    Utiliser torch.testing.assert_close avec tolérances explicites.

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
def test_tensor_memory_equals_numel_times_element_size():
    """
    Objectif
    --------
    Calculer la mémoire brute occupée par un tensor dense.

    Concepts à comprendre
    ---------------------
    - mémoire tensor
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/tensors/memory.py

    Comportement à vérifier
    -----------------------
    La mémoire théorique vaut numel * element_size.

    Assertion attendue
    ------------------
    assert bytes == x.numel() * x.element_size()

    Hints d'implémentation
    ----------------------
    Utiliser numel(), element_size().

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


