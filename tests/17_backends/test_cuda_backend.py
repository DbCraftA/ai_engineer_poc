"""Section 17.3 — le backend CUDA : même opération, autre matériel, mêmes valeurs.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/17_backends/test_cuda_backend.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(shape (4, 8), float32, `"cuda"`, rtol=1e-4 / atol=1e-5) : c'est volontaire. Un test doit
énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

C'est le premier test de portabilité réel du chapitre : la référence vient du backend CPU
de 17.2, l'exécution vient du GPU. Deux matériels différents ne rendent pas des bits
identiques, donc la comparaison se fait avec une tolérance annoncée — et il faut
synchroniser avant de lire, sinon on lit un buffer que le GPU n'a pas fini d'écrire. Les
markers `gpu` / `cuda` laissent `tests/conftest.py` sauter ce test sur une machine sans GPU.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices. Même raison pour l'import
# du module cible : c'est ton code d'`Act` qui l'appellera, le linter le voit donc inutilisé.
# ruff: noqa: F401, F821

import pytest
import torch


@pytest.mark.tdd
@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_backend_matches_cpu_reference():
    """Roadmap 17.3 — le GPU doit retomber sur la référence CPU, à une tolérance près.

    Objectif d'apprentissage
    ------------------------
    « Le GPU donne un autre résultat » n'est presque jamais un bug : c'est de l'arithmétique
    flottante. Une réduction sur K éléments n'est pas associative en float32, et le GPU la
    découpe en blocs parallèles puis somme les blocs, dans un ordre qui n'est pas celui du
    CPU. Sur Ampere et au-delà s'ajoute TF32, qui tronque la mantisse des entrées d'un
    matmul float32 à 10 bits (désactivé par défaut pour `matmul` dans PyTorch, mais
    activable pour gagner en vitesse). D'où la règle de tout le dépôt : on ne compare jamais
    deux matériels avec `torch.equal`, on annonce une tolérance et on la justifie. rtol=1e-4
    et atol=1e-5 en float32 laissent passer un ordre de réduction différent, mais pas une
    erreur de transposition, de dtype ou d'indices.

    Deuxième piège, propre au GPU : les kernels sont asynchrones. Lire les valeurs sans
    `synchronize()` peut fonctionner par chance sur un petit tenseur et échouer sous charge.
    Le backend expose donc la synchronisation, et le code du modèle l'appelle sans savoir
    qu'elle est vide sur CPU.

    Schéma mental
    -------------
        a (4, 8) float32 "cpu"   b (8, 8) float32 "cpu"      (seq=4, hidden=8, projection 8x8)

        cpu_backend.matmul(a, b)                  -> reference   (4, 8) float32 "cpu"
        cuda_backend.matmul(a', b') puis sync     -> cuda_result (4, 8) float32 "cuda:0"

        reference vs cuda_result.cpu()  ->  proches, PAS identiques bit à bit
            |diff| <= atol + rtol x |reference| = 1e-5 + 1e-4 x |reference|

    Ce que ce test vérifie
    ----------------------
    1. le backend CUDA s'annonce comme « cuda » et sait se synchroniser ;
    2. son résultat vit en VRAM avec la shape (4, 8) et le dtype float32 de la référence,
       donc sur un device différent de celle-ci ;
    3. après synchronisation et rapatriement, les valeurs concordent avec la référence CPU
       à rtol=1e-4 / atol=1e-5 ;
    4. la comparaison exige un rapatriement explicite, et la référence CPU n'a pas bougé.

    API à faire émerger (cible proposée : `src/inference_lab/backends/cuda.py`)
    ------------------------------------------------------------------------
        class CudaBackend:
            name: str = "cuda"
            device: torch.device

            def to_device(self, tensor: torch.Tensor) -> torch.Tensor: ...
            def synchronize(self) -> None: ...
            def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...

        La roadmap indique « future backends » sans chemin. Réutilise l'existant plutôt que
        de le dupliquer : `resolve_device` de `backends/devices.py` (6.1) pour choisir le
        device, `to_device` de `backends/transfers.py` (6.5) pour les transferts.

    Indice : `torch.cuda.synchronize()` attend la fin des kernels du device courant.
    `same_device` compare les devices INDEX COMPRIS : `torch.device("cuda")` (index `None`)
    n'est pas égal à `torch.device("cuda:0")`, d'où la comparaison sur `.type` quand seul le
    type de mémoire compte. Piège : `torch.testing.assert_close` refuse deux tenseurs de
    devices différents, il faut rapatrier avant de comparer.
    """

    pytest.skip("Roadmap TDD 17.3 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.backends.cpu import CpuBackend
    from inference_lab.backends.cuda import CudaBackend
    from inference_lab.backends.devices import same_device
    from inference_lab.backends.transfers import to_device

    # Arrange — pas de garde `if torch.cuda.is_available()` ici : le hook de `tests/conftest.py`
    #           l'a déjà faite via le marker `cuda`. Construire `a`, de shape (4, 8), et `b`,
    #           de shape (8, 8), en `torch.float32` sur CPU, déterministes (seed fixée) et à
    #           valeurs non entières, pour qu'un écart d'arrondi soit possible. Instancier
    #           `cpu_backend` et `cuda_backend`.

    # Act — calculer `reference` avec `cpu_backend.matmul(a, b)`, envoyer `a` et `b` sur le
    #       device de `cuda_backend` avec `to_device`, appeler `cuda_backend.matmul` dans
    #       `cuda_result`, SYNCHRONISER, puis rapatrier sur CPU dans `cuda_result_on_cpu`.

    # Assert 1 — le backend CUDA s'annonce et sait attendre la fin de ses kernels
    assert cuda_backend.name == "cuda"
    assert cuda_backend.device.type == "cuda"
    assert cuda_backend.synchronize() is None

    # Assert 2 — le résultat vit en VRAM, avec la shape et le dtype de la référence
    assert cuda_result.device.type == "cuda"
    assert cuda_result.shape == (4, 8)
    assert cuda_result.dtype is torch.float32
    assert same_device(cuda_result, reference) is False

    # Assert 3 — même opération, mêmes valeurs, à la tolérance float32 annoncée
    torch.testing.assert_close(cuda_result_on_cpu, reference, rtol=1e-4, atol=1e-5)

    # Assert 4 — la comparaison passe par un rapatriement explicite, la référence est intacte
    assert cuda_result_on_cpu.device.type == "cpu"
    assert reference.device.type == "cpu"
    assert cuda_result_on_cpu.shape == reference.shape
    assert cuda_result_on_cpu.dtype is reference.dtype
