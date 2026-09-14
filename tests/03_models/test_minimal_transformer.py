import pytest


@pytest.mark.tdd
@pytest.mark.model
def test_minimal_transformer_stacks_requested_number_of_blocks():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_minimal_transformer_stacks_requested_number_of_blocks`.

    Concepts à comprendre
    ---------------------
    - modèle complet
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/models/minimal_transformer/model.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 3.3.
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
@pytest.mark.model
def test_model_forward_outputs_logits_for_each_token_and_vocabulary_entry():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_model_forward_outputs_logits_for_each_token_and_vocabulary_entry`.

    Concepts à comprendre
    ---------------------
    - forward
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/models/minimal_transformer/model.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 3.4.
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
@pytest.mark.model
def test_lm_head_projects_hidden_dimension_to_vocabulary_size():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_lm_head_projects_hidden_dimension_to_vocabulary_size`.

    Concepts à comprendre
    ---------------------
    - LM Head
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/models/minimal_transformer/model.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 3.5.
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


