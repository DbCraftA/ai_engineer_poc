"""Section 9.1 — l'environnement Triton : compilateur, GPU CUDA et skip automatique.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/09_triton/test_triton_environment.py`.
              Le test DOIT échouer : les variables de la partie `Arrange` n'existent pas
              encore (`NameError`) — ici il n'y a aucun module `src/` à créer.
2. GREEN    : écrire le minimum de code dans le test, juste assez pour faire passer les
              assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(version majeure >= 3, noms exacts des markers, chaînes exactes de `conftest.py`) : c'est
volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule
que le code testé.

Ce test ouvre le chapitre Triton, mais il n'écrit aucun kernel : il vérifie la plomberie.
Un kernel Triton n'est pas du Python interprété, c'est du code compilé pour un GPU précis :
sans compilateur Triton ET sans device CUDA, les tests 9.2 à 9.5 ne peuvent pas s'exécuter,
et doivent alors *sauter* proprement plutôt que d'échouer. Rappel de cap pour tout le
chapitre : on n'écrit pas un kernel pour « aller plus vite », on l'écrit pour répondre à une
question mémoire/kernel identifiée — et la seule chose qu'un test valide ici, c'est la
CORRECTION du kernel face à PyTorch, jamais sa performance.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

from pathlib import Path

import pytest


@pytest.mark.tdd
@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.triton
def test_triton_tests_skip_when_triton_or_cuda_is_unavailable():
    """Roadmap 9.1 — un test de kernel ne s'exécute que si le compilateur ET le GPU sont là.

    Objectif d'apprentissage
    ------------------------
    Les sections 1 à 5 tournent partout, y compris sur un portable sans GPU. À partir de
    Triton, ce n'est plus vrai : le kernel est compilé en PTX pour l'architecture du device
    présent. Une suite de tests utilisable doit donc distinguer trois situations : « le
    matériel manque » (skip, la suite reste verte), « le matériel est là et le kernel est
    faux » (échec réel, celui qui nous intéresse) et « le marker n'est pas déclaré » (bug
    silencieux : pytest se contente d'un avertissement et `pytest -m triton` ne filtre plus
    rien). Ce test verrouille la première et la troisième situation, pour que les échecs de
    9.2 à 9.5 ne parlent plus que de numérique.

    Schéma mental
    -------------
        pyproject.toml : markers = [..., "triton: requires Triton and CUDA"]
                         [project.optional-dependencies] gpu = ["triton>=3.0; ..."]
                  |
        @pytest.mark.triton  ->  markers du test = {tdd, gpu, cuda, triton}
                  |
        tests/conftest.py, pytest_runtest_setup :
            find_spec("triton") is None       -> skip « Triton is not installed »
            not torch.cuda.is_available()     -> skip « Triton tests require CUDA »
            sinon                             -> le corps du test s'exécute vraiment

        ici : find_spec("triton") -> ModuleSpec, device_count >= 1, version majeure >= 3

    Ce que ce test vérifie
    ----------------------
    1. le hook de `conftest.py` a laissé passer : Triton est détectable et CUDA disponible,
       avec au moins un device visible ;
    2. Triton est réellement importable et sa version majeure vaut au moins 3, comme le
       déclare l'extra `gpu` de `pyproject.toml` ;
    3. ce test porte bien les markers `gpu`, `cuda` et `triton`, et le marker `triton` est
       déclaré dans `pyproject.toml` : `pytest -m triton` sélectionne donc ce chapitre ;
    4. la décision de sauter reste centralisée dans `tests/conftest.py`, qui teste à la fois
       la présence du paquet et celle de CUDA (aucune garde locale à écrire dans les tests).

    Cible roadmap : « infrastructure » — aucun module `src/` à écrire
    ---------------------------------------------------------------
    Ce test exerce `torch`, `importlib.util`, les markers et la configuration du dépôt. Il
    doit passer au vert dès que le `pytest.skip` est retiré et que les variables d'`Arrange`
    existent.

    Indice : `from importlib.util import find_spec` puis `find_spec("triton")` renvoie `None`
    si le paquet est absent, sans lever d'exception (c'est exactement ce que fait
    `conftest.py`). Les markers d'une fonction de test se relisent sur son attribut
    `pytestmark`, une liste d'objets `Mark` ayant chacun un `.name`. Piège : ne remets pas de
    `if not torch.cuda.is_available(): pytest.skip(...)` dans le corps — ce serait dupliquer
    le hook, et l'assert 1 ne prouverait plus rien.
    """

    pytest.skip("Roadmap TDD 9.1 — supprimer cette ligne pour démarrer le cycle RED")

    # Arrange — créer `triton_spec` en interrogeant `importlib.util.find_spec` sur le paquet
    #           `"triton"`, `cuda_available` et `device_count` en interrogeant CUDA,
    #           `triton_version` en lisant le `__version__` du module `triton` importé,
    #           `test_markers`, l'ensemble des noms de markers portés par CE test (lus sur son
    #           attribut `pytestmark`), et `repo_root`, la racine du dépôt déduite de
    #           `__file__`.

    # Act — lire le contenu texte de `pyproject.toml` dans `pyproject_source` et celui de
    #       `tests/conftest.py` dans `conftest_source`.

    # Assert 1 — le hook de conftest a laissé passer : compilateur et GPU sont présents
    assert triton_spec is not None
    assert cuda_available is True
    assert device_count >= 1

    # Assert 2 — Triton est importable, dans la version exigée par l'extra `gpu`
    assert int(triton_version.split(".")[0]) >= 3
    assert "triton>=3.0" in pyproject_source

    # Assert 3 — les markers de ce test rendent `pytest -m triton` opérant
    assert {"gpu", "cuda", "triton"}.issubset(test_markers)
    assert "triton: requires Triton and CUDA" in pyproject_source

    # Assert 4 — le skip matériel est centralisé, et teste bien les DEUX conditions
    assert "def pytest_runtest_setup" in conftest_source
    assert 'find_spec("triton")' in conftest_source
    assert "torch.cuda.is_available()" in conftest_source
    assert Path(repo_root, "tests", "conftest.py").is_file()
