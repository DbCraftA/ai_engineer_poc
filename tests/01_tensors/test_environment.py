import pytest


@pytest.mark.tdd
def test_pytorch_is_available():
    """
    Objectif
    --------
    Vérifier que le socle PyTorch est importable avant toute expérience.

    Concepts à comprendre
    ---------------------
    - environment, CPU/GPU, disponibilité CUDA
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    infrastructure uniquement

    Comportement à vérifier
    -----------------------
    Importer torch et lire sa version sans exception.

    Assertion attendue
    ------------------
    assert torch.__version__ is not None

    Hints d'implémentation
    ----------------------
    Utiliser import torch et éventuellement packaging.version pour comparer les versions.

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
@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_tests_are_skipped_when_cuda_is_unavailable():
    """
    Objectif
    --------
    Garantir que la suite CPU reste verte même sans GPU NVIDIA.

    Concepts à comprendre
    ---------------------
    - environment, CPU/GPU, disponibilité CUDA
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    tests/conftest.py

    Comportement à vérifier
    -----------------------
    Un test marqué cuda/gpu ne doit pas échouer quand torch.cuda.is_available() vaut False.

    Assertion attendue
    ------------------
    assert pytest.skip est déclenché via tests/conftest.py

    Hints d'implémentation
    ----------------------
    Utiliser pytest_runtest_setup, item.keywords, torch.cuda.is_available().

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


