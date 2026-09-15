"""Section 17.5 — découverte des capacités : déclarer ce que le matériel sait faire.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/17_backends/test_capability_discovery.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(l'ensemble {float32, float16, bfloat16}, `True` / `False`, `NotImplementedError`) : c'est
volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule
que le code testé.

Une capacité manquante doit se voir. Le pire comportement possible pour un backend est la
dégradation silencieuse : accepter du float8 et calculer en float32, ou ignorer une demande
de FlashAttention. Le benchmark comparerait alors deux matériels qui n'ont pas fait le même
travail, et la conclusion serait fausse sans qu'aucun test n'échoue.

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
def test_backend_reports_supported_dtypes_and_operations():
    """Roadmap 17.5 — une capacité absente est refusée, jamais dégradée en silence.

    Objectif d'apprentissage
    ------------------------
    Le moteur doit pouvoir demander au matériel « sais-tu faire ça ? » avant d'essayer, pour
    choisir un dtype de poids, activer une attention spécialisée ou refuser une
    configuration. C'est ce qui permet d'écrire une politique unique — « prends bfloat16 si
    le backend le déclare, sinon float32 » — au lieu de tester le nom du matériel. Deux
    exigences en découlent. D'abord la déclaration doit être un ensemble FINI et explicite,
    pas un « oui à tout ». Ensuite déclaration et exécution doivent concorder : ce qui est
    déclaré s'exécute vraiment, et ce qui n'est pas déclaré lève une erreur claire.

    Le contre-exemple à retenir : un backend qui accepte du float8 en le promouvant en
    float32 « pour que ça marche ». Le calcul réussit, la mémoire annoncée est fausse, le
    benchmark mesure autre chose que ce qu'on croit, et rien ne casse. `NotImplementedError`
    avec le dtype fautif dans le message coûte trois lignes et évite cette classe de bugs.

    Schéma mental
    -------------
        CpuBackend().supported_dtypes -> {float32, float16, bfloat16}   (3 dtypes, fini)

            supports("float32")         -> True         supports("fp8")             -> False
            supports("bfloat16")        -> True         supports("flash_attention") -> False
            supports("matmul")          -> True

        matmul(a bf16 (4, 8), b bf16 (8, 8))  -> (4, 8) bfloat16     déclaré  -> exécuté
        matmul(a fp8  (4, 8), b fp8  (8, 8))  -> NotImplementedError  non déclaré -> refusé

    Ce que ce test vérifie
    ----------------------
    1. les dtypes supportés forment un ensemble explicite et fini, dont float8 est exclu ;
    2. `supports` répond par un vrai booléen, aussi bien sur un dtype que sur une opération ;
    3. déclaration et exécution concordent : un dtype déclaré s'exécute et le résultat
       conserve ce dtype, sans promotion cachée ;
    4. un dtype non déclaré lève `NotImplementedError`, et le message nomme le dtype fautif.

    API à faire émerger (cible proposée : `src/inference_lab/backends/cpu.py`, contrat
    déclaré dans `src/inference_lab/backends/base.py`)
    -------------------------------------------------------------------------------
        class CpuBackend:
            supported_dtypes: frozenset[torch.dtype]

            def supports(self, capability: str) -> bool: ...
            def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...

        `capability` est un nom : soit un nom de dtype (`"float32"`, `"bfloat16"`, `"fp8"`),
        soit un nom d'opération (`"matmul"`, `"flash_attention"`). La roadmap indique
        « future backends » sans chemin ; on reste dans le package `backends/` existant.

    Indice : `torch.empty(2, 2, dtype=torch.float8_e4m3fn)` s'alloue sans problème sur CPU —
    c'est le CALCUL qui n'est pas supporté, d'où l'intérêt de refuser explicitement. Pour le
    message d'erreur, `str(torch.float8_e4m3fn)` vaut `"torch.float8_e4m3fn"` et contient
    donc `"float8"`. Piège : ne construis pas `supported_dtypes` en testant chaque dtype à
    l'exécution ; c'est une DÉCLARATION, écrite à la main, que ce test confronte ensuite au
    comportement réel.
    """

    pytest.skip("Roadmap TDD 17.5 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.backends.cpu import CpuBackend

    # Arrange — instancier `backend` (`CpuBackend()`). Construire `a_bf16` (4, 8) et `b_bf16`
    #           (8, 8) en `torch.bfloat16` sur CPU, un dtype DÉCLARÉ supporté, valeurs
    #           déterministes ; puis `a_fp8` et `b_fp8`, mêmes shapes en
    #           `torch.float8_e4m3fn`, un dtype volontairement NON déclaré.

    # Act — lire `backend.supported_dtypes`, interroger `backend.supports(...)` sur plusieurs
    #       dtypes et opérations, exécuter `backend.matmul` sur le couple bfloat16 dans
    #       `bf16_result`, et garder le couple float8 pour l'assertion de refus.

    # Assert 1 — la déclaration est un ensemble explicite et fini
    assert set(backend.supported_dtypes) == {torch.float32, torch.float16, torch.bfloat16}
    assert torch.float8_e4m3fn not in backend.supported_dtypes

    # Assert 2 — `supports` rend un booléen, sur les dtypes comme sur les opérations
    assert backend.supports("float32") is True
    assert backend.supports("bfloat16") is True
    assert backend.supports("matmul") is True
    assert backend.supports("fp8") is False
    assert backend.supports("flash_attention") is False

    # Assert 3 — ce qui est déclaré s'exécute, sans promotion cachée du dtype
    assert bf16_result.dtype is torch.bfloat16
    assert bf16_result.shape == (4, 8)
    assert bf16_result.device.type == "cpu"

    # Assert 4 — ce qui n'est pas déclaré est refusé explicitement, message compris
    with pytest.raises(NotImplementedError) as refusal:
        _ = backend.matmul(a_fp8, b_fp8)
    assert "float8" in str(refusal.value)
