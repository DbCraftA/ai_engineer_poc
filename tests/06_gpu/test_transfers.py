"""Section 6.5 — transferts CPU -> GPU -> CPU : changer de mémoire sans changer de valeurs.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/06_gpu/test_transfers.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(shape (4, 8), 32 éléments, float32, égalité BIT À BIT) : c'est volontaire. Un test doit
énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Un transfert est une COPIE d'octets sur le bus PCIe : il ne convertit rien, ne réordonne
rien, ne perd rien. C'est ce qui autorise la stratégie de tout le dépôt — calculer une
référence sur CPU, exécuter sur GPU, comparer les deux — et c'est aussi le coût qu'il faut
apprendre à éviter dans la boucle de decode, où chaque `.cpu()` sur un logit sérialise le
GPU et le CPU.

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
def test_tensor_transfer_preserves_values():
    """Roadmap 6.5 — un aller-retour CPU / GPU change le device, jamais les valeurs.

    Objectif d'apprentissage
    ------------------------
    Le transfert est le seul endroit du pipeline où les octets traversent le PCIe. Deux
    conséquences à retenir : côté correction, la copie est exacte, donc une divergence
    numérique entre CPU et GPU ne vient JAMAIS du transfert mais du calcul (ordre de
    réduction, TF32, dtype) ; côté performance, un transfert coûte de la latence, d'où les
    règles pratiques du dépôt — charger les poids une fois, garder le KV cache en VRAM, et
    ne rapatrier qu'un token à la fois en decode, pas les logits complets.

    Schéma mental
    -------------
        x (4, 8) float32 sur "cpu"    32 éléments x 4 octets = 128 octets

            --.to("cuda")-->  gpu_copy (4, 8) float32 sur "cuda:0"   (copie PCIe)
            --.cpu()------->  back_on_cpu (4, 8) float32 sur "cpu"

        torch.equal(back_on_cpu, x) is True   <- égalité bit à bit, pas une tolérance
        data_ptr différents                   <- trois buffers distincts, pas des vues

    Ce que ce test vérifie
    ----------------------
    1. la copie GPU annonce `device.type == "cuda"` et garde la shape (4, 8) et float32 ;
    2. l'aller-retour revient sur CPU et rend un tenseur strictement égal à l'original
       (`torch.equal`, aucune tolérance) ;
    3. un transfert est une copie et non une vue : les buffers diffèrent, et un transfert
       vers le device courant ne copie rien du tout ;
    4. rien n'a été promu ni tronqué : 32 éléments, float32, même somme des deux côtés.

    API à faire émerger (cible proposée : `src/inference_lab/backends/transfers.py`)
    ------------------------------------------------------------------------------
        def to_device(tensor: torch.Tensor, device: torch.device | str) -> torch.Tensor: ...

        La roadmap indique « backend » sans chemin : `backends/` regroupe déjà tout ce qui
        dépend du matériel (cf. 6.1).

    Indice : `tensor.to(device)` suffit et renvoie l'objet lui-même si le tenseur est déjà
    sur le bon device et dans le bon dtype — d'où l'assert `is`. Pièges : `.to()` ne modifie
    jamais le tenseur en place (il faut réaffecter), et `non_blocking=True` n'accélère que
    depuis une mémoire *pinned*, sinon il ne fait rien de plus.
    """

    pytest.skip("Roadmap TDD 6.5 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.backends.transfers import to_device

    # Arrange — créer `cpu_tensor`, un tenseur de shape (4, 8) en `torch.float32` sur CPU,
    #           déterministe (seed fixée) et à valeurs non entières, pour que l'égalité bit à
    #           bit ait un sens. Prévoir `device`, le device GPU visé.

    # Act — envoyer `cpu_tensor` sur le GPU dans `gpu_copy`, puis le rapatrier dans
    #       `back_on_cpu`, en passant à chaque fois par l'API cible.

    # Assert 1 — la copie vit sur le GPU, avec la même shape et le même dtype
    assert gpu_copy.device.type == "cuda"
    assert gpu_copy.shape == (4, 8)
    assert gpu_copy.dtype is torch.float32

    # Assert 2 — l'aller-retour est exact, sans tolérance
    assert back_on_cpu.device.type == "cpu"
    assert torch.equal(back_on_cpu, cpu_tensor)

    # Assert 3 — un transfert copie, un transfert inutile ne copie pas
    assert gpu_copy.data_ptr() != cpu_tensor.data_ptr()
    assert back_on_cpu.data_ptr() != gpu_copy.data_ptr()
    assert to_device(cpu_tensor, "cpu") is cpu_tensor

    # Assert 4 — aucune promotion ni troncature au passage
    assert back_on_cpu.numel() == 32
    assert back_on_cpu.dtype is torch.float32
    assert back_on_cpu.sum().item() == cpu_tensor.sum().item()
    assert cpu_tensor.device.type == "cpu"
