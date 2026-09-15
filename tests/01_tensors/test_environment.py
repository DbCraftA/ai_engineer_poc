"""Section 1.1 — environnement d'exécution : PyTorch, CPU, CUDA et skip matériel.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/01_tensors/test_environment.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(4 éléments, somme 6.0, major >= 2) : c'est volontaire. Un test doit énoncer la vérité
attendue, pas la recalculer avec la même formule que le code testé.

Exception de cette section : la roadmap indique « infrastructure uniquement » et
`tests/conftest.py` comme cibles. Il n'y a donc AUCUN module `src/` à écrire ici : ces
deux tests valident l'environnement lui-même (version de PyTorch, présence de CUDA) et
le hook `pytest_runtest_setup` de `tests/conftest.py` qui saute automatiquement les tests
marqués `gpu`, `cuda`, `triton` ou `distributed` quand la machine ne peut pas les exécuter.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_pytorch_is_available():
    """Roadmap 1.1 — savoir sur quel matériel le laboratoire va réellement tourner.

    Objectif d'apprentissage
    ------------------------
    Tout le reste de la roadmap (attention, KV cache, kernels Triton, mesures de
    bande passante) dépend de deux informations connues dès le départ : quelle version
    de PyTorch est installée, et si un GPU CUDA est visible. Un test d'attention peut
    tourner sur CPU ; un benchmark de décodage mémoire-bound n'a de sens que sur GPU.
    Ce test est le point d'entrée qui rend cette information explicite au lieu de la
    découvrir par une erreur au milieu d'un notebook.

    Schéma mental
    -------------
        torch.__version__          -> "2.x.y"  (le dépôt exige torch >= 2.3)
        torch.cuda.is_available()  -> True | False  (jamais None, jamais une chaîne)
        device = "cuda" si disponible, sinon "cpu"

        tenseur 1D des 4 premiers entiers naturels (float32) : numel = 4, somme = 6.0

    Ce que ce test vérifie
    ----------------------
    1. PyTorch est importable et annonce une version majeure >= 2 ;
    2. `torch.cuda.is_available()` renvoie un vrai booléen, utilisable dans un `if` ;
    3. le device retenu est `"cuda"` si CUDA est disponible, sinon `"cpu"` ;
    4. un calcul minimal sur CPU donne le résultat attendu : 0 + 1 + 2 + 3 = 6.0.

    Cible roadmap : « infrastructure uniquement » — aucun module `src/` à écrire
    --------------------------------------------------------------------------
    Ce test doit passer au vert dès que le `pytest.skip` est retiré et que les variables
    d'`Arrange` existent : il ne teste que l'environnement, pas du code du dépôt.

    Indice : `torch.__version__` est une chaîne du type `"2.4.1+cpu"` ; découpe-la sur
    `"."` pour en extraire l'entier majeur. Piège classique : `torch.cuda.is_available()`
    ne lève pas d'exception sur une machine sans GPU, elle renvoie simplement `False` —
    c'est une valeur à tester, pas un plantage à éviter.
    """

    pytest.skip("Roadmap TDD 1.1 — supprimer cette ligne pour démarrer le cycle RED")

    # Arrange — créer `version_major`, la version majeure de PyTorch sous forme d'entier,
    #           `cuda_available`, le résultat brut de l'interrogation de CUDA, et `x`, un
    #           tenseur 1D `torch.float32` contenant les 4 premiers entiers naturels
    #           (à construire avec `torch.arange`, sans écrire les valeurs à la main).

    # Act — déduire de `cuda_available` la chaîne `device` à utiliser pour les calculs, puis
    #       réduire `x` par une somme pour vérifier que le moteur de calcul répond.

    # Assert 1 — la version installée respecte la contrainte du dépôt
    assert version_major >= 2

    # Assert 2 — la disponibilité de CUDA est un booléen, pas une valeur ambiguë
    assert isinstance(cuda_available, bool)
    assert cuda_available == torch.cuda.is_available()

    # Assert 3 — le device retenu découle directement de cette disponibilité
    assert device in {"cpu", "cuda"}
    assert device == ("cuda" if cuda_available else "cpu")

    # Assert 4 — le calcul minimal donne la valeur attendue, écrite en dur
    assert x.numel() == 4
    assert x.dtype is torch.float32
    assert x.sum().item() == 6.0


@pytest.mark.tdd
@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_tests_are_skipped_when_cuda_is_unavailable():
    """Roadmap 1.1 — un test marqué `cuda` ne s'exécute que si CUDA existe vraiment.

    Objectif d'apprentissage
    ------------------------
    La roadmap mélange des tests portables (CPU) et des tests qui n'ont de sens que sur
    GPU (kernels Triton, mesure de bande passante HBM, occupation mémoire du KV cache).
    Plutôt que de parsemer le code de `if torch.cuda.is_available()`, le dépôt centralise
    la décision dans `tests/conftest.py::pytest_runtest_setup` : le hook lit les markers
    du test et appelle `pytest.skip` lui-même. Conséquence, à retenir pour tout le reste
    du projet : le corps d'un test marqué `cuda` peut supposer le GPU présent, sans garde.

    Schéma mental
    -------------
        markers du test : {tdd, gpu, cuda}
                              |
        conftest.pytest_runtest_setup
                              |
            torch.cuda.is_available() ?
                  /                     \\
              False                     True
          skip AVANT le corps       le corps s'exécute
                                    -> cuda_available == True
                                    -> device_count >= 1

        tenseur 1D de 4 valeurs float32 (0..3) : CPU -> GPU -> CPU, somme 6.0 conservée

    Ce que ce test vérifie
    ----------------------
    1. si ce corps s'exécute, c'est que le hook de `conftest.py` a laissé passer :
       CUDA est donc réellement disponible ;
    2. au moins un GPU est visible depuis le processus ;
    3. un tenseur copié sur le GPU annonce `device.type == "cuda"`, et l'aller-retour
       le ramène bien sur `"cpu"` ;
    4. le transfert ne change aucune valeur : la somme reste 6.0.

    Cible roadmap : `tests/conftest.py` — aucun module `src/` à écrire
    -----------------------------------------------------------------
    Ce test documente et exerce le mécanisme de skip existant. Sur une machine sans GPU,
    le résultat attendu n'est pas `passed` mais `skipped` : c'est la preuve que le hook
    fonctionne.

    Indice : `torch.cuda.device_count()` donne le nombre de GPU visibles, `tensor.to("cuda")`
    et `tensor.cpu()` font les transferts. Piège : ne mets pas de `if
    torch.cuda.is_available()` dans le corps, ce serait dupliquer le rôle du hook — c'est
    justement ce que ce test cherche à rendre inutile.
    """

    pytest.skip("Roadmap TDD 1.1 — supprimer cette ligne pour démarrer le cycle RED")

    # Arrange — créer `cuda_available` et `device_count` en interrogeant CUDA, puis `x`,
    #           un tenseur 1D `torch.float32` des 4 premiers entiers naturels, alloué sur CPU.

    # Act — copier `x` sur le GPU dans `gpu_tensor`, puis le ramener sur CPU dans
    #       `back_on_cpu`, pour observer un aller-retour complet entre les deux mémoires.

    # Assert 1 — le hook de conftest n'a pas skipé : CUDA est bien là
    assert cuda_available is True

    # Assert 2 — au moins un GPU est visible dans ce processus
    assert device_count >= 1

    # Assert 3 — chaque tenseur sait sur quelle mémoire il vit
    assert gpu_tensor.device.type == "cuda"
    assert back_on_cpu.device.type == "cpu"

    # Assert 4 — le transfert est fidèle : mêmes valeurs de part et d'autre
    assert back_on_cpu.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert gpu_tensor.sum().item() == 6.0
