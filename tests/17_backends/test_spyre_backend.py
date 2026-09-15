"""Section 17.4 — le backend Spyre : le vrai test de portabilité de l'architecture.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/17_backends/test_spyre_backend.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(shape (4, 8), float32, `"spyre"`, rtol=1e-3 / atol=1e-4) : c'est volontaire. Un test doit
énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Ce fichier est un SQUELETTE assumé : Spyre n'est pas installé sur la machine de
développement, donc le test se saute proprement après avoir importé son module cible. Deux
skips se superposent et ne disent pas la même chose : celui du cycle TDD (« section pas
encore travaillée », à supprimer par l'apprenant) puis le skip conditionnel matériel
(« backend absent ici », à conserver pour toujours). Un test sauté est honnête ; un test qui
se rabat en silence sur le CPU serait un faux vert et détruirait l'intérêt du chapitre.

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
def test_spyre_backend_matches_reference_for_supported_operation():
    """Roadmap 17.4 — un matériel absent se saute, il ne se simule pas.

    Objectif d'apprentissage
    ------------------------
    CUDA valide la portabilité à moitié seulement : PyTorch y expose la même API qu'en CPU,
    donc une abstraction bancale peut y survivre. Spyre est le vrai juge, parce qu'il n'a ni
    les mêmes kernels, ni le même modèle de mémoire, ni la même attention. Si le contrat de
    17.1 est bon, ajouter un troisième backend ne touche PAS le modèle Qwen : on écrit
    `spyre.py`, on l'enregistre, et 17.6 le compare automatiquement à la référence CPU.

    L'autre leçon est une leçon de discipline de test : sur une machine sans le matériel, la
    bonne réponse est `skipped`, jamais `passed`. On saute donc sur une SONDE de
    disponibilité (`importlib.util.find_spec` sur le SDK vendeur, sans import réel ni effet
    de bord), pas sur un `try / except ImportError` qui avalerait aussi une vraie erreur de
    configuration. La tolérance est plus large qu'en 17.3 : un accélérateur peut accumuler un
    matmul float32 dans une précision interne réduite, ce qui déplace le résultat davantage
    qu'un simple changement d'ordre de réduction.

    Schéma mental
    -------------
        spyre_is_available() is False  ->  pytest.skip(...)  ->  test "skipped", pas "passed"
        spyre_is_available() is True   ->  on exécute la MÊME propriété que 17.3 :

            a (4, 8) float32 "cpu"  b (8, 8) float32 "cpu"

            cpu_backend.matmul(a, b)     -> reference (4, 8) float32 "cpu"
            spyre_backend.matmul(a', b') -> résultat sur le device Spyre
                                            puis rapatrié -> (4, 8) float32 "cpu"

            |diff| <= 1e-4 + 1e-3 x |reference|

    Ce que ce test vérifie
    ----------------------
    1. le backend s'annonce comme « spyre » et déclare supporter l'opération demandée
       avant qu'on la lui demande ;
    2. rapatrié sur CPU, son résultat a bien la shape (4, 8) et le dtype float32 de la
       référence ;
    3. la même propriété qu'en 17.3 : accord numérique avec la référence CPU, ici à
       rtol=1e-3 / atol=1e-4 ;
    4. pas de faux vert : la sonde rend un vrai booléen et le backend testé n'est pas le
       backend CPU déguisé.

    API à faire émerger (cible proposée : `src/inference_lab/backends/spyre.py`)
    -------------------------------------------------------------------------
        def spyre_is_available() -> bool: ...

        class SpyreBackend:
            name: str = "spyre"
            device: torch.device
            supported_dtypes: frozenset[torch.dtype]

            def supports(self, capability: str) -> bool: ...
            def to_device(self, tensor: torch.Tensor) -> torch.Tensor: ...
            def synchronize(self) -> None: ...
            def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...

        La roadmap indique « future backends » sans chemin : `spyre.py` complète `cpu.py`
        (17.2) et `cuda.py` (17.3) dans le package `backends/` existant.

    Indice : `spyre_is_available()` se réduit à
    `importlib.util.find_spec("<paquet du SDK>") is not None` — pour la pile AIU / Spyre
    d'IBM, le module PyTorch se nomme aujourd'hui `torch_sendnn`. L'équivalent en une ligne
    est `pytest.importorskip("torch_sendnn")`, mais on garde la sonde nommée pour que le
    registre de 17.6 puisse l'interroger sans passer par `pytest`. Piège : ne fabrique
    surtout pas un `SpyreBackend` de secours qui calcule sur CPU — la classe doit rester
    inutilisable tant que le matériel est absent.
    """

    pytest.skip("Roadmap TDD 17.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.backends.cpu import CpuBackend
    from inference_lab.backends.spyre import SpyreBackend, spyre_is_available

    # Skip conditionnel matériel — à conserver même après le cycle RED : Spyre n'est pas
    # installé sur la machine de développement. La sonde est interrogée APRÈS l'import du
    # module cible, pour que le premier RED reste bien un `ModuleNotFoundError`.
    if not spyre_is_available():
        pytest.skip("Backend Spyre indisponible sur cette machine : rien à vérifier ici.")

    # Arrange — construire `a`, de shape (4, 8), et `b`, de shape (8, 8), en `torch.float32`
    #           sur CPU, déterministes (seed fixée) et à valeurs non entières. Instancier
    #           `cpu_backend` et `spyre_backend`.

    # Act — calculer `reference` avec `cpu_backend.matmul(a, b)`, envoyer `a` et `b` sur le
    #       device de `spyre_backend` via sa méthode `to_device`, appeler
    #       `spyre_backend.matmul`, SYNCHRONISER, puis rapatrier le résultat sur CPU dans
    #       `spyre_result_on_cpu`.

    # Assert 1 — le backend s'annonce et déclare la capacité avant qu'on l'exerce
    assert spyre_backend.name == "spyre"
    assert spyre_backend.supports("matmul") is True
    assert torch.float32 in spyre_backend.supported_dtypes

    # Assert 2 — rapatrié sur CPU, le résultat est comparable à la référence
    assert spyre_result_on_cpu.device.type == "cpu"
    assert spyre_result_on_cpu.shape == (4, 8)
    assert spyre_result_on_cpu.dtype is torch.float32

    # Assert 3 — même propriété qu'en 17.3 : accord avec l'oracle CPU, tolérance annoncée
    torch.testing.assert_close(spyre_result_on_cpu, reference, rtol=1e-3, atol=1e-4)

    # Assert 4 — pas de faux vert : sonde booléenne, et ce n'est pas le backend CPU déguisé
    assert spyre_is_available() is True
    assert spyre_backend.name != cpu_backend.name
