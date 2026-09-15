"""Section 17.6 — cross-hardware : tous les backends disponibles, une seule vérité.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/17_backends/test_cross_hardware.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(shape (4, 8), float32, rtol 1e-5 en fp32, 1e-3 en fp16, 1e-2 en bf16) : c'est volontaire.
Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code
testé.

Ce test clôt le chapitre en énonçant la promesse du projet (cf. README, « Long-term goal ») :
même modèle, mêmes poids, même prompt, matériels différents, résultats comparables. Il ne
suppose PAS un matériel donné : il découvre les backends utilisables via un registre, et
doit donc rester vert avec un seul backend disponible comme avec trois.

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
def test_supported_backends_produce_numerically_compatible_results():
    """Roadmap 17.6 — un registre découvre le matériel présent, et tous doivent concorder.

    Objectif d'apprentissage
    ------------------------
    Un test de portabilité écrit en dur pour CPU + CUDA + Spyre est ingérable : il échoue sur
    le laptop, échoue en CI sans GPU, et il faut le réécrire à chaque nouveau matériel. La
    bonne forme est la DÉCOUVERTE : le registre n'instancie que ce qui est réellement
    utilisable ici (CPU toujours, CUDA si un GPU répond, Spyre si son SDK est installé), et
    le test boucle sur ce qu'il a trouvé. Avec un seul backend, il vérifie encore quelque
    chose de vrai — que la référence est cohérente avec elle-même ; avec trois, il devient le
    filet de sécurité de tout le chapitre. C'est le même mécanisme que le hook
    `pytest_runtest_setup` de `tests/conftest.py`, mais côté code de production.

    Deuxième idée : la tolérance est une propriété du DTYPE, pas du backend. Elle mesure le
    nombre de bits de mantisse disponibles, donc l'erreur d'arrondi acceptable dans une
    réduction, pas la marque du matériel. Elle vit donc dans une table, une fois pour tout le
    dépôt, plutôt que recopiée à la main dans chaque test :

        float32  : 24 bits de mantisse -> rtol 1e-5
        float16  : 11 bits de mantisse -> rtol 1e-3
        bfloat16 :  8 bits de mantisse -> rtol 1e-2

    Schéma mental
    -------------
        available_backends()  ->  (CpuBackend(),)                 machine sans GPU
                              ->  (CpuBackend(), CudaBackend())   cette machine
                              ->  (CpuBackend(), CudaBackend(), SpyreBackend())

        a (4, 8) float32   b (8, 8) float32   ->  reference = résultat du backend "cpu"

        pour chaque backend : matmul, synchronize, rapatriement sur CPU
            results = {"cpu": (4, 8), "cuda": (4, 8), ...}

        |result - reference| <= 1e-5 + 1e-5 x |reference|   pour CHAQUE backend
        results["cpu"] == reference                          bit à bit, c'est l'oracle

    Ce que ce test vérifie
    ----------------------
    1. le registre ne renvoie que des backends utilisables, dont le CPU toujours, et rien
       d'inconnu ;
    2. il y a exactement un résultat par backend découvert, tous ramenés dans la même forme
       comparable : shape (4, 8), float32, sur CPU ;
    3. chaque backend disponible concorde avec la référence CPU à la tolérance float32
       annoncée, et le CPU lui est exactement égal, bit à bit ;
    4. la tolérance dépend du dtype et non du backend, et se resserre quand la mantisse
       s'allonge.

    API à faire émerger (cible proposée : `src/inference_lab/backends/registry.py`)
    -----------------------------------------------------------------------------
        RELATIVE_TOLERANCES: dict[torch.dtype, float]

        def available_backends() -> tuple[Backend, ...]: ...

        La roadmap indique « future backends » sans chemin. `registry.py` s'appuie sur
        `cpu.py` (17.2), `cuda.py` (17.3), `spyre.py` (17.4) et interroge
        `torch.cuda.is_available()` ainsi que `spyre_is_available()` pour décider qui
        instancier.

    Indice : construis `results` avec une compréhension de dictionnaire indexée par
    `backend.name`, et rapatrie chaque résultat avec `to_device(..., "cpu")` de
    `backends/transfers.py` (6.5) avant toute comparaison. Pièges : `assert_close` refuse
    deux devices différents, et le CPU doit apparaître dans `results` comme les autres —
    ne le traite pas à part, sinon le test ne prouve plus que la boucle fonctionne.
    """

    pytest.skip("Roadmap TDD 17.6 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.backends.registry import RELATIVE_TOLERANCES, available_backends
    from inference_lab.backends.transfers import to_device

    # Arrange — appeler `available_backends()` dans `backends`. Construire `a`, de shape
    #           (4, 8), et `b`, de shape (8, 8), en `torch.float32` sur CPU, déterministes
    #           (seed fixée) et à valeurs non entières. Calculer `reference` avec le backend
    #           dont le `name` vaut "cpu".

    # Act — pour chaque backend de `backends` : envoyer `a` et `b` sur son device, appeler
    #       `matmul`, synchroniser, rapatrier le résultat sur CPU, et ranger le tout dans
    #       `results`, un dict `{nom du backend: tenseur rapatrié}`.

    # Assert 1 — le registre ne promet que ce qui existe ici, et le CPU est toujours présent
    assert len(backends) >= 1
    assert "cpu" in results
    assert set(results) <= {"cpu", "cuda", "spyre"}
    assert set(results) == {backend.name for backend in backends}

    # Assert 2 — un résultat par backend, tous ramenés dans la même forme comparable
    assert len(results) == len(backends)
    assert all(result.shape == (4, 8) for result in results.values())
    assert all(result.dtype is torch.float32 for result in results.values())
    assert all(result.device.type == "cpu" for result in results.values())

    # Assert 3 — chaque matériel disponible concorde avec l'oracle CPU, qui s'égale exactement
    for name, result in sorted(results.items()):
        torch.testing.assert_close(
            result, reference, rtol=RELATIVE_TOLERANCES[torch.float32], atol=1e-5, msg=name
        )
    assert torch.equal(results["cpu"], reference)

    # Assert 4 — la tolérance est fonction du dtype, pas du backend
    assert RELATIVE_TOLERANCES[torch.float32] == 1e-5
    assert RELATIVE_TOLERANCES[torch.float16] == 1e-3
    assert RELATIVE_TOLERANCES[torch.bfloat16] == 1e-2
