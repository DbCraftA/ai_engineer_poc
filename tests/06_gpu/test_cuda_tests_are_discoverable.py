"""Infrastructure (hors roadmap) — sélection par marker : `pytest -m cuda` doit fonctionner.

Statut particulier de ce fichier
--------------------------------
Ce fichier n'est PAS un exercice de la roadmap : c'est le test de plomberie du chapitre 6,
déjà vert. Il n'a donc pas de `pytest.skip` d'activation et son corps est écrit en entier.
Le cycle TDD (RED -> GREEN -> REFACTOR) décrit dans les autres fichiers de tests ne
s'applique pas ici : il n'y a aucun module `src/` à faire émerger.

Ce qu'il protège : un test GPU doit rester *sélectionnable* par son marker (`pytest -m cuda`
avant un benchmark, `pytest -m "not gpu"` en CI sans GPU), ce marker doit être déclaré dans
`pyproject.toml` — sinon pytest n'émet qu'un avertissement et la sélection ne filtre rien,
bug silencieux classique — et l'allocation en VRAM doit fonctionner réellement quand le hook
de `tests/conftest.py` laisse passer le test.

Lecture du test : `Arrange` prépare les données, `Act` exécute, `Assert` compare aux valeurs
attendues écrites en dur (shape (4,), somme 4.0, noms de markers exacts).

Roadmap et modules cibles des autres fichiers : `tests/ROADMAP.md`.
"""

from pathlib import Path

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_tensor_allocation_is_skipped_when_cuda_is_unavailable():
    """Infrastructure — un test GPU est sélectionnable par marker et saute sans GPU.

    Objectif d'apprentissage
    ------------------------
    La suite du dépôt mélange des tests portables et des tests matériels. Pour rester
    utilisable, elle doit permettre trois usages : tout lancer (les tests GPU sautent si
    besoin), ne lancer que le GPU (`pytest -m cuda`, avant un benchmark), ou l'exclure
    (`pytest -m "not gpu"` en CI sans GPU). Ces trois usages reposent sur deux conditions
    concrètes : le test porte les bons markers, et ces markers sont déclarés dans
    `pyproject.toml`. Un marker non déclaré ne provoque pas d'erreur, juste un
    avertissement — et une sélection qui ne filtre rien, bug silencieux classique.

    Schéma mental
    -------------
        pyproject.toml [tool.pytest.ini_options] markers = ["gpu: ...", "cuda: ...", ...]
                  |
        @pytest.mark.gpu @pytest.mark.cuda  ->  markers du test = {tdd, gpu, cuda}
                  |                                     |
        pytest -m cuda        -> sélectionné    pytest -m "not gpu" -> désélectionné
                  |
        conftest.pytest_runtest_setup : skip si CUDA absent, sinon le corps s'exécute

        torch.ones(4) sur "cuda" : shape (4,), float32, somme = 4.0

    Ce que ce test vérifie
    ----------------------
    1. le corps ne s'exécute que GPU présent : CUDA disponible, au moins un device visible ;
    2. l'allocation en VRAM marche : shape (4,), float32, device `"cuda"`, somme 4.0 ;
    3. ce test porte bien les markers `gpu` et `cuda`, donc `pytest -m cuda` le sélectionne ;
    4. ces markers sont déclarés dans `pyproject.toml`, et le skip matériel reste centralisé
       dans `tests/conftest.py` (aucune garde locale nécessaire dans ce corps).

    Cible : infrastructure uniquement — aucun module `src/` à écrire
    ---------------------------------------------------------------
    Ce test exerce `torch`, les markers et la configuration du dépôt. Il est déjà vert et
    reste actif : il n'a pas de ligne `pytest.skip` d'activation.

    À savoir : les markers d'une fonction de test se relisent sur son attribut `pytestmark`
    (une liste d'objets `Mark`, chacun avec un `.name`). Piège : ne remets pas de
    `if not torch.cuda.is_available(): pytest.skip(...)` dans le corps — ce serait dupliquer
    le hook de `conftest.py`, et l'assert 1 ne prouverait plus rien.
    """

    # Arrange — état de CUDA, un tenseur en VRAM, les markers de ce test, la racine du dépôt
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    gpu_tensor = torch.ones(4, dtype=torch.float32, device="cuda")
    test_markers = {
        mark.name
        for mark in test_cuda_tensor_allocation_is_skipped_when_cuda_is_unavailable.pytestmark
    }
    repo_root = Path(__file__).resolve().parents[2]

    # Act — relire la configuration du dépôt et forcer un calcul réel sur le GPU
    pyproject_source = Path(repo_root, "pyproject.toml").read_text(encoding="utf-8")
    conftest_source = Path(repo_root, "tests", "conftest.py").read_text(encoding="utf-8")
    gpu_sum = gpu_tensor.sum().item()

    # Assert 1 — le hook de conftest a laissé passer : le GPU est présent
    assert cuda_available is True
    assert device_count >= 1

    # Assert 2 — l'allocation en VRAM et le calcul donnent les valeurs attendues
    assert gpu_tensor.device.type == "cuda"
    assert gpu_tensor.shape == (4,)
    assert gpu_tensor.dtype is torch.float32
    assert gpu_sum == 4.0

    # Assert 3 — les markers portés par ce test rendent `-m cuda` opérant
    assert {"gpu", "cuda"}.issubset(test_markers)

    # Assert 4 — markers déclarés et skip matériel centralisé
    assert "gpu: requires GPU" in pyproject_source
    assert "cuda: requires CUDA" in pyproject_source
    assert "def pytest_runtest_setup" in conftest_source
    assert Path(repo_root, "pyproject.toml").is_file()
