import pytest


@pytest.mark.tdd
@pytest.mark.model
def test_greedy_sampling_selects_highest_logit():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_greedy_sampling_selects_highest_logit`.

    Concepts à comprendre
    ---------------------
    - greedy sampling
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/inference/sampling.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 3.6.
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
def test_temperature_changes_probability_distribution():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_temperature_changes_probability_distribution`.

    Concepts à comprendre
    ---------------------
    - temperature
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/inference/sampling.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 3.7.
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
def test_top_k_excludes_tokens_outside_k_highest_logits():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_top_k_excludes_tokens_outside_k_highest_logits`.

    Concepts à comprendre
    ---------------------
    - top-k
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/inference/sampling.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 3.8.
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
def test_top_p_limits_candidates_by_cumulative_probability():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_top_p_limits_candidates_by_cumulative_probability`.

    Concepts à comprendre
    ---------------------
    - top-p
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/inference/sampling.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 3.9.
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


