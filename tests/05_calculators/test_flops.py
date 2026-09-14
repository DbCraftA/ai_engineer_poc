import pytest


@pytest.mark.tdd
def test_linear_layer_flops_are_estimated_from_matrix_dimensions():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_linear_layer_flops_are_estimated_from_matrix_dimensions`.

    Concepts à comprendre
    ---------------------
    - FLOPs linear
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/calculators/flops.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 5.5.
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
def test_model_decode_flops_can_be_estimated_from_architecture():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_model_decode_flops_can_be_estimated_from_architecture`.

    Concepts à comprendre
    ---------------------
    - FLOPs modèle
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    calculator

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 5.6.
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


