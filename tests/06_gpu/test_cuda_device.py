"""Section 6.1 — device CUDA : où vit un tenseur et pourquoi ça change tout.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/06_gpu/test_cuda_device.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(shape (2, 3), 6 éléments, somme 0.0, `"cuda"` / `"cpu"`) : c'est volontaire. Un test doit
énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Cette section installe la notion de *device* : à partir d'ici, un tenseur n'est plus
seulement une shape et un dtype, il a aussi une adresse mémoire physique (RAM hôte ou
VRAM du GPU). Tout le reste du chapitre 6 (transferts, mesure du temps, mémoire allouée)
en dépend, et les tests marqués `gpu` / `cuda` sont sautés automatiquement par le hook
`pytest_runtest_setup` de `tests/conftest.py` quand la machine n'a pas de GPU.

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
def test_tensor_can_be_created_on_cuda():
    """Roadmap 6.1 — un tenseur est alloué sur un device précis, choisi explicitement.

    Objectif d'apprentissage
    ------------------------
    Un moteur d'inférence ne « tourne pas sur GPU » globalement : chaque tenseur est
    alloué quelque part. Les poids de Qwen2.5-0.5B, le KV cache et les activations
    doivent vivre dans la VRAM, sinon chaque couche paierait un aller-retour PCIe qui
    coûte plus cher que le calcul lui-même. Le laboratoire a donc besoin d'un seul
    endroit qui répond à la question « sur quel device travaille-t-on ? », au lieu de
    semer des `torch.cuda.is_available()` dans tout le code.

    Schéma mental
    -------------
        resolve_device(prefer_cuda=True)  -> device(type="cuda", index=0)   # GPU présent
        resolve_device(prefer_cuda=False) -> device(type="cpu")             # forcé CPU

        torch.zeros(2, 3, dtype=float32, device=cuda:0)
            shape (2, 3) = 6 éléments, somme 0.0, mémoire = 6 x 4 = 24 octets de VRAM

        la shape et le dtype ne changent pas en traversant le PCIe ; seul le device change

    Ce que ce test vérifie
    ----------------------
    1. le device résolu est le GPU quand CUDA est disponible, et le CPU reste joignable
       à la demande ;
    2. le tenseur créé annonce `device.type == "cuda"` et porte l'index du GPU courant ;
    3. l'allocation sur GPU conserve exactement la shape (2, 3) et le dtype float32 ;
    4. le calcul a bien lieu sur le GPU : la réduction rend 0.0 et reste sur `"cuda"`.

    API à faire émerger (cible proposée : `src/inference_lab/backends/devices.py`)
    ----------------------------------------------------------------------------
        def resolve_device(prefer_cuda: bool = True) -> torch.device: ...

        La roadmap indique « helpers GPU » sans chemin : `backends/` est le package
        cohérent pour tout ce qui dépend du matériel.

    Indice : `torch.device("cuda")` n'a pas d'index (`index is None`), alors qu'un tenseur
    alloué dessus reçoit l'index du GPU courant (`cuda:0`). Compare donc toujours
    `tensor.device.type`, jamais `tensor.device == torch.device("cuda")`, qui est faux.
    """

    pytest.skip("Roadmap TDD 6.1 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.backends.devices import resolve_device

    # Arrange — pas de garde `if torch.cuda.is_available()` ici : le hook de `conftest.py`
    #           l'a déjà faite. Créer `device` en demandant à l'API le device de travail
    #           préféré, puis `gpu_tensor`, un tenseur de shape (2, 3) en `torch.float32`
    #           alloué directement sur ce `device` et rempli de zéros (pas de `.to(...)`
    #           après coup : on veut une allocation d'emblée en VRAM).

    # Act — interroger l'API une seconde fois en refusant le GPU, pour vérifier qu'elle
    #       sait aussi désigner le CPU.

    # Assert 1 — le device résolu est le GPU, et le CPU reste accessible à la demande
    assert device.type == "cuda"
    assert resolve_device(prefer_cuda=False).type == "cpu"

    # Assert 2 — le tenseur vit sur le GPU courant, index compris
    assert gpu_tensor.device.type == "cuda"
    assert gpu_tensor.device.index == torch.cuda.current_device()
    assert gpu_tensor.is_cuda is True

    # Assert 3 — changer de mémoire ne change ni la shape ni le dtype
    assert gpu_tensor.shape == (2, 3)
    assert gpu_tensor.dtype is torch.float32

    # Assert 4 — le calcul a réellement lieu sur le GPU, valeurs attendues en dur
    assert gpu_tensor.numel() == 6
    assert gpu_tensor.sum().item() == 0.0
    assert gpu_tensor.sum().device.type == "cuda"


@pytest.mark.tdd
@pytest.mark.gpu
@pytest.mark.cuda
def test_cpu_and_cuda_tensors_report_different_devices():
    """Roadmap 6.1 — deux tenseurs de devices différents ne se combinent pas.

    Objectif d'apprentissage
    ------------------------
    PyTorch refuse d'opérer entre RAM hôte et VRAM : il n'insère jamais de copie
    implicite, parce qu'un transfert PCIe caché ruinerait silencieusement les
    performances. C'est l'erreur runtime la plus fréquente quand on branche un tokenizer
    (qui rend des tenseurs CPU) sur un modèle chargé en VRAM, ou un masque causal
    construit sur CPU sur des scores d'attention calculés sur GPU. Retenir la règle :
    on déplace explicitement, ou ça lève.

    Schéma mental
    -------------
        cpu_tensor (2, 3) float32 sur "cpu"      gpu_tensor (2, 3) float32 sur "cuda:0"

        cpu_tensor + gpu_tensor  -> RuntimeError: Expected all tensors to be on the
                                    same device, but found at least two devices,
                                    cuda:0 and cpu!

        cpu_tensor.to("cuda") + gpu_tensor  -> OK, résultat sur "cuda"

    Ce que ce test vérifie
    ----------------------
    1. chaque tenseur annonce son device et les deux devices sont différents ;
    2. `same_device(a, b)` nomme cette comparaison et rend un vrai booléen ;
    3. une opération mixte CPU / GPU lève `RuntimeError`, et le message parle de device ;
    4. après déplacement explicite sur le même device, l'opération passe et les valeurs
       sont restées identiques de part et d'autre du PCIe.

    API à faire émerger (cible proposée : `src/inference_lab/backends/devices.py`)
    ----------------------------------------------------------------------------
        def same_device(a: torch.Tensor, b: torch.Tensor) -> bool: ...

    Indice : compare `a.device == b.device` (le device complet, index inclus) et renvoie
    un `bool`, pas un `torch.Tensor`. Piège : `torch.equal(cpu_tensor, gpu_tensor)` lève
    lui aussi `RuntimeError` au lieu de rendre `False` — ce n'est donc pas un moyen de
    tester l'égalité entre deux devices, il faut d'abord rapatrier l'un des deux.
    """

    pytest.skip("Roadmap TDD 6.1 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.backends.devices import resolve_device, same_device

    # Arrange — créer `cpu_tensor`, un tenseur de shape (2, 3) en `torch.float32` resté sur
    #           CPU, aux valeurs déterministes et non nulles (seed fixée ou valeurs
    #           construites), puis `gpu_tensor`, sa copie envoyée sur le device rendu par
    #           `resolve_device()`.

    # Act — demander à l'API si ces deux tenseurs partagent le même device, puis tenter
    #       volontairement une addition entre les deux mémoires.

    # Assert 1 — chaque tenseur sait où il vit, et les deux devices diffèrent
    assert cpu_tensor.device.type == "cpu"
    assert gpu_tensor.device.type == "cuda"
    assert cpu_tensor.device != gpu_tensor.device

    # Assert 2 — l'API nomme la comparaison et rend un booléen
    assert same_device(cpu_tensor, gpu_tensor) is False
    assert same_device(gpu_tensor, gpu_tensor) is True
    assert same_device(cpu_tensor, cpu_tensor) is True

    # Assert 3 — PyTorch refuse l'opération mixte, sans copie implicite
    with pytest.raises(RuntimeError) as mixed_device_error:
        _ = cpu_tensor + gpu_tensor
    assert "device" in str(mixed_device_error.value)

    # Assert 4 — le déplacement explicite débloque l'opération, sans changer les valeurs
    assert same_device(cpu_tensor.to(gpu_tensor.device), gpu_tensor) is True
    assert (cpu_tensor.to(gpu_tensor.device) + gpu_tensor).device.type == "cuda"
    assert torch.equal(gpu_tensor.cpu(), cpu_tensor)
