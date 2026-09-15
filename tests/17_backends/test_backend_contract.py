"""Section 17.1 — le contrat d'un backend : ce que tout matériel doit savoir faire.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/17_backends/test_backend_contract.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(`"cpu"`, la liste des sept membres du contrat, `True` / `False`) : c'est volontaire. Un test
doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Le chapitre 17 a un seul objectif : la PORTABILITÉ. Le même modèle Qwen doit tourner sur
CPU, CUDA et Spyre sans que son code ne parle de matériel. Ce premier test fixe donc la
frontière : un backend encapsule device, dtypes, synchronisation et opération lourde, et
c'est le backend CPU — toujours disponible — qui sert de référence pour vérifier le contrat.

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
def test_backend_exposes_required_tensor_and_execution_capabilities():
    """Roadmap 17.1 — un backend est un contrat explicite, pas une suite de `if device`.

    Objectif d'apprentissage
    ------------------------
    Sans contrat, la portabilité se paie en `if device == "cuda": ... else: ...` semés dans
    l'attention, le KV cache, le chargement des poids et la boucle de decode. Chaque nouveau
    matériel (Spyre) oblige alors à relire tout le moteur, et chaque oubli devient un bug
    silencieux. Un backend rassemble ces différences en UN objet qui répond toujours aux
    mêmes questions : comment tu t'appelles, où calcules-tu, quels dtypes acceptes-tu,
    comment te synchroniser, sais-tu faire cette opération. Le code du modèle, lui, ne
    connaît plus que ce contrat.

    Attention à la frontière (cf. README, « Le backend ne doit PAS abstraire ») : on
    n'encapsule pas chaque addition PyTorch. On encapsule le device, le dtype, la
    compilation, la synchronisation et les opérations lourdes spécialisées.

    Schéma mental
    -------------
        CpuBackend()
            .name              -> "cpu"
            .device            -> device(type="cpu")
            .supported_dtypes  -> {float32, float16, bfloat16}
            .supports("matmul")         -> True
            .supports("flash_attention")-> False
            .to_device(t (2, 3))        -> tenseur (2, 3) sur .device
            .synchronize()              -> None
            .matmul(a, b)               -> tenseur

        7 membres, 3 backends (cpu / cuda / spyre), 1 seule interface pour le modèle

    Ce que ce test vérifie
    ----------------------
    1. le backend sait se nommer et dire sur quel device il calcule ;
    2. la liste des membres exigés est déclarée UNE fois dans `base.py`, et le backend CPU
       les possède tous ;
    3. la partie exécutable du contrat est réellement appelable, et `synchronize()` est
       licite même là où elle n'a rien à faire ;
    4. les capacités sont des booléens explicites, et float32 est le socle commun ;
       `to_device` place bien le tenseur sur le device du backend.

    API à faire émerger (cible proposée : `src/inference_lab/backends/base.py` et
    `src/inference_lab/backends/cpu.py`)
    -----------------------------------------------------------------------------
        REQUIRED_BACKEND_MEMBERS: tuple[str, ...]

        class Backend(Protocol):
            name: str
            device: torch.device
            supported_dtypes: frozenset[torch.dtype]

            def supports(self, capability: str) -> bool: ...
            def to_device(self, tensor: torch.Tensor) -> torch.Tensor: ...
            def synchronize(self) -> None: ...
            def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...

        class CpuBackend: ...

        La roadmap indique « future backends » sans chemin : `backends/` existe déjà et
        contient `devices.py` (6.1) et `transfers.py` (6.5). `CpuBackend.to_device` doit
        déléguer à `inference_lab.backends.transfers.to_device` au lieu de réécrire `.to()`.

    Indice : on teste le contrat avec `hasattr` et `callable`, pas avec
    `isinstance(backend, Backend)` : un `Protocol` n'est vérifiable à l'exécution que s'il
    est décoré `@runtime_checkable`, et même là il ne contrôle que les noms, pas les
    signatures. Le tuple `REQUIRED_BACKEND_MEMBERS` documente donc le contrat de façon
    testable et réutilisable pour les autres backends.
    """

    pytest.skip("Roadmap TDD 17.1 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.backends.base import REQUIRED_BACKEND_MEMBERS
    from inference_lab.backends.cpu import CpuBackend

    # Arrange — instancier `backend`, le backend CPU (`CpuBackend()`), seule implémentation
    #           toujours disponible et donc référence du chapitre. Créer aussi `probe`, un
    #           tenseur de shape (2, 3) en `torch.float32` sur CPU, pour vérifier que
    #           `to_device` rend un tenseur sur le device annoncé par le backend.

    # Act — lire l'identité du backend (`name`, `device`, `supported_dtypes`), puis appeler
    #       la partie exécutable du contrat : `supports`, `to_device`, `synchronize`.

    # Assert 1 — un backend sait se nommer et dire où il calcule
    assert isinstance(backend.name, str)
    assert backend.name == "cpu"
    assert backend.device.type == "cpu"

    # Assert 2 — le contrat est déclaré une fois pour toutes, pas dispersé dans le moteur
    assert REQUIRED_BACKEND_MEMBERS == (
        "name",
        "device",
        "supported_dtypes",
        "supports",
        "to_device",
        "synchronize",
        "matmul",
    )
    assert all(hasattr(backend, member) for member in REQUIRED_BACKEND_MEMBERS)

    # Assert 3 — la partie exécutable du contrat est appelable, synchroniser est toujours licite
    assert callable(backend.supports)
    assert callable(backend.to_device)
    assert callable(backend.synchronize)
    assert callable(backend.matmul)
    assert backend.synchronize() is None

    # Assert 4 — capacités booléennes explicites, float32 comme socle commun
    assert backend.supports("float32") is True
    assert backend.supports("matmul") is True
    assert backend.supports("flash_attention") is False
    assert torch.float32 in backend.supported_dtypes
    assert backend.to_device(probe).device == backend.device
