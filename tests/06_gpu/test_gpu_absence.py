"""Section 6.8 — absence de GPU : les tests marqués `gpu` doivent skipper, pas échouer.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/06_gpu/test_gpu_absence.py -k <nom_du_test>`.
              Le test DOIT échouer : les variables de la partie `Arrange` n'existent pas
              encore (`NameError`) — ici il n'y a aucun module `src/` à créer.
2. GREEN    : écrire le minimum de code dans le test, juste assez pour faire passer les
              assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(`True` / `False` pour chaque combinaison de markers, noms de markers exacts) : c'est
volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même
formule que le code testé.

La cible de cette section est `tests/conftest.py`, pas `src/` : elle documente le contrat
d'exécution de tout le dépôt. Un dépôt pédagogique doit rester clonable sur un portable
sans GPU — les tests matériels y apparaissent alors `skipped`, jamais `failed`, et la suite
reste verte. C'est la même mécanique qui protégera les tests Triton (section 9) et
distribués (section 16).

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

from pathlib import Path

import pytest
import torch


@pytest.mark.tdd
@pytest.mark.gpu
@pytest.mark.cuda
def test_gpu_marked_tests_skip_cleanly_without_cuda():
    """Roadmap 6.8 — le hook `pytest_runtest_setup` saute les tests matériels impossibles.

    Objectif d'apprentissage
    ------------------------
    Deux stratégies existent pour un test qui exige du matériel : la garde locale
    (`if not torch.cuda.is_available(): pytest.skip(...)`, répétée dans chaque test) ou la
    décision centralisée par marker. Le dépôt a choisi la seconde : `tests/conftest.py`
    implémente `pytest_runtest_setup(item)`, lit `item.keywords` et appelle `pytest.skip`
    AVANT le corps du test. Conséquence pratique pour toute la roadmap : un test marqué
    `gpu` ou `cuda` peut supposer le GPU présent sans aucune garde, et la sélection
    `pytest -m "not gpu"` reste possible sur une machine sans CUDA.

    Schéma mental
    -------------
        markers du test : {tdd, gpu, cuda}
                  |
        conftest.pytest_runtest_setup(item)
                  |
        "gpu" ou "cuda" dans item.keywords  et  torch.cuda.is_available() == False
                  |                                        |
                 oui -> pytest.skip(...)                  non -> le corps s'exécute

        règle attendue :
            ({"gpu"},  cuda=False) -> skip      ({"gpu", "cuda"}, cuda=True) -> exécute
            ({"cuda"}, cuda=False) -> skip      ({"cpu"},         cuda=False) -> exécute

    Ce que ce test vérifie
    ----------------------
    1. si ce corps s'exécute, le hook a laissé passer : CUDA est réellement disponible et
       au moins un GPU est visible ;
    2. la règle de décision du hook, énoncée cas par cas : `gpu` ou `cuda` sans CUDA
       donnent un skip, et rien d'autre ne provoque de skip ;
    3. `tests/conftest.py` contient bien ce hook, appuyé sur `torch.cuda.is_available()`
       et sur `pytest.skip` ;
    4. les markers `gpu` et `cuda` sont déclarés dans `pyproject.toml` et portés par ce
       test : la sélection `-m gpu` fonctionne sans avertissement de marker inconnu.

    Cible roadmap : `tests/conftest.py` — aucun module `src/` à écrire
    -----------------------------------------------------------------
    Ce test décrit et verrouille un mécanisme qui EXISTE déjà. Sur une machine sans GPU, le
    résultat attendu n'est pas `passed` mais `skipped` : c'est précisément la preuve que le
    hook fait son travail.

    Indice : écris dans l'`Arrange` une petite fonction pure qui reproduit la règle du hook
    (elle prend un ensemble de markers et un booléen, et renvoie un booléen) — c'est le
    moyen de tester la logique sans relancer pytest dans pytest. Les sources se lisent avec
    `Path(__file__).resolve().parents[2]` puis `read_text(encoding="utf-8")`, et les markers
    d'un test se relisent sur la fonction elle-même via son attribut `pytestmark`.
    """

    pytest.skip("Roadmap TDD 6.8 — supprimer cette ligne pour démarrer le cycle RED")

    # Arrange — écrire `should_skip_for_hardware(keywords, cuda_available)`, une fonction pure
    #           locale qui renvoie `True` quand le marker `gpu` ou `cuda` est présent alors que
    #           CUDA est absent, et `False` sinon : c'est la règle appliquée par
    #           `tests/conftest.py`. Créer aussi `repo_root`, la racine du dépôt déduite de
    #           `__file__`, puis `conftest_source` et `pyproject_source`, le contenu texte de
    #           `tests/conftest.py` et de `pyproject.toml`. Enfin `test_markers`, l'ensemble des
    #           noms de markers portés par ce test, lu sur son attribut `pytestmark`.

    # Act — appliquer la règle aux quatre combinaisons de markers du schéma mental, et relever
    #       la disponibilité réelle de CUDA dans `cuda_available` ainsi que le nombre de GPU
    #       visibles dans `device_count`.

    # Assert 1 — le hook n'a pas skipé ce test : le GPU est bien là
    assert cuda_available is True
    assert cuda_available == torch.cuda.is_available()
    assert device_count >= 1

    # Assert 2 — la règle de décision, cas par cas
    assert should_skip_for_hardware({"tdd", "gpu"}, cuda_available=False) is True
    assert should_skip_for_hardware({"tdd", "cuda"}, cuda_available=False) is True
    assert should_skip_for_hardware({"tdd", "gpu", "cuda"}, cuda_available=True) is False
    assert should_skip_for_hardware({"tdd", "cpu"}, cuda_available=False) is False

    # Assert 3 — le hook existe vraiment dans `tests/conftest.py`
    assert "def pytest_runtest_setup" in conftest_source
    assert "torch.cuda.is_available()" in conftest_source
    assert "pytest.skip" in conftest_source

    # Assert 4 — les markers sont déclarés et portés par ce test
    assert "gpu: requires GPU" in pyproject_source
    assert "cuda: requires CUDA" in pyproject_source
    assert {"gpu", "cuda"}.issubset(test_markers)
    assert Path(repo_root, "tests", "conftest.py").is_file()
