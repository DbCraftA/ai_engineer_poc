import pytest


@pytest.mark.tdd
@pytest.mark.perf
def test_benchmark_executes_warmup_before_measurement():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_benchmark_executes_warmup_before_measurement`.

    Concepts à comprendre
    ---------------------
    - benchmark warmup
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/benchmarks/runner.py

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 7.1.
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
@pytest.mark.perf
def test_benchmark_collects_multiple_measurements():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_benchmark_collects_multiple_measurements`.

    Concepts à comprendre
    ---------------------
    - répétitions
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    runner

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 7.2.
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
@pytest.mark.perf
def test_benchmark_reports_median_latency():
    """
    Objectif
    --------
    Démontrer progressivement le comportement exprimé par `test_benchmark_reports_median_latency`.

    Concepts à comprendre
    ---------------------
    - médiane
    - shapes et layout lorsque pertinent
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    runner

    Comportement attendu
    --------------------
    Cette specification TDD décrit le comportement attendu pour la section 7.3.
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


