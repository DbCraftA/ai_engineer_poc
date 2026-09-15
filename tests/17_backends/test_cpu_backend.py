"""Section 17.2 — le backend CPU : l'oracle numérique de tous les autres backends.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/17_backends/test_cpu_backend.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(shape (2, 4), 8 éléments, chaque case à 3.0, somme 24.0) : c'est volontaire. Un test doit
énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Le backend CPU n'est pas là pour être rapide : il est là pour être VRAI. C'est lui qui
tourne partout, sans GPU ni SDK vendeur, donc c'est lui qui fournit la référence à laquelle
17.3 (CUDA), 17.4 (Spyre) et 17.6 (cross-hardware) se comparent. S'il est faux, tout le
chapitre compare des résultats à un mensonge.

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
def test_cpu_backend_executes_reference_operation():
    """Roadmap 17.2 — le backend toujours disponible exécute l'opération de référence.

    Objectif d'apprentissage
    ------------------------
    Toute la stratégie de validation du dépôt repose sur une exécution CPU de référence :
    on calcule la vérité en float32 sur CPU, puis on vérifie que le GPU ou l'accélérateur
    retombe dessus à une tolérance près. C'est exactement ce que fait `tests/qwen` face à
    Hugging Face. Le backend CPU doit donc être ennuyeux et transparent : il n'optimise
    rien, ne change ni le dtype ni le device, et ne fait rien de plus que le PyTorch
    générique qu'il enveloppe. `synchronize()` existe quand même, car le modèle appelle le
    contrat sans savoir sur quel matériel il tourne — sur CPU c'est simplement une
    opération vide.

    Schéma mental
    -------------
        a (2, 3) float32 remplie de 1     b (3, 4) float32 remplie de 1

            matmul  ->  result (2, 4) float32 sur "cpu"

        chaque case = 1x1 + 1x1 + 1x1 = 3.0   (la dimension réduite K vaut 3)
        8 cases à 3.0  ->  somme = 24.0

    Ce que ce test vérifie
    ----------------------
    1. le résultat combine les shapes en (2, 4), garde float32 et reste sur CPU ;
    2. les valeurs attendues, calculées à la main : 3.0 partout, somme 24.0, et
       exactement, sans tolérance (des sommes de 1.0 sont exactes en binaire) ;
    3. l'oracle : le backend CPU rend bit à bit le même tenseur que `a @ b` ;
    4. le backend s'annonce comme « cpu » et sa synchronisation est une opération vide.

    API à faire émerger (cible proposée : `src/inference_lab/backends/cpu.py`)
    ------------------------------------------------------------------------
        class CpuBackend:
            name: str = "cpu"
            device: torch.device

            def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
            def synchronize(self) -> None: ...

        La roadmap indique « future backends » sans chemin : `cpu.py` dans le package
        `backends/` existant, aux côtés de `devices.py` (6.1) et `transfers.py` (6.5).

    Indice : `torch.matmul(a, b)` (ou `a @ b`) suffit, et `synchronize` se contente de
    `return None` — le CPU est synchrone par construction, il n'y a pas de file d'attente
    à vider. Piège : ne convertis pas les entrées « au cas où » (pas de `.float()`
    défensif), sinon ce backend cesse d'être un oracle et masque les erreurs de dtype.
    """

    pytest.skip("Roadmap TDD 17.2 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.backends.cpu import CpuBackend

    # Arrange — instancier `backend` (`CpuBackend()`), puis construire `a`, de shape (2, 3),
    #           et `b`, de shape (3, 4), tous deux en `torch.float32` sur CPU et remplis de
    #           UNS : aucune valeur aléatoire ici, on veut un résultat calculable de tête.

    # Act — exécuter l'opération de référence du chapitre dans `result` via
    #       `backend.matmul(a, b)`, puis synchroniser le backend avant de lire les valeurs.

    # Assert 1 — shape combinée, dtype de référence, mémoire hôte
    assert result.shape == (2, 4)
    assert result.numel() == 8
    assert result.dtype is torch.float32
    assert result.device.type == "cpu"

    # Assert 2 — les valeurs attendues, écrites en dur et exactes
    torch.testing.assert_close(result, torch.full((2, 4), 3.0), rtol=0.0, atol=0.0)
    assert result.sum().item() == 24.0

    # Assert 3 — l'oracle : rien de plus que le PyTorch générique de référence
    assert torch.equal(result, a @ b)

    # Assert 4 — identité du backend et synchronisation vide mais licite
    assert backend.name == "cpu"
    assert backend.synchronize() is None
