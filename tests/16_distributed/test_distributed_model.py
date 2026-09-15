"""Section 16.6 — bout en bout : un modèle « tensor parallel » simulé égale le mono-device.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/16_distributed/test_distributed_model.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(logits (1, 4, 32), 4 AllReduce pour 2 blocs, tolérance fp32 rtol=1e-5 / atol=1e-6) : c'est
volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule que
le code testé.

Dimensions jouets constantes de toute la section 16, complétées pour un modèle entier :
hidden=8, intermediate=16, num_heads=4, head_dim=2, seq=4, batch=1, tensor_parallel_size=2,
vocab=32, num_layers=2.

Pourquoi ce test tourne sur UN SEUL processus
--------------------------------------------
Le marker `distributed` est conservé, mais il n'y a ni
`torch.distributed.init_process_group`, ni NCCL, ni second GPU — l'assert 1 le vérifie
explicitement. On SIMULE le parallélisme de tenseurs : les poids sont découpés en shards (un
par « device »), chaque contribution locale est calculée, puis recombinée à la main par les
collectives pures de la section 16.4 / 16.5. Ce que ce test prouve est la seule chose qui
puisse être fausse dans une implémentation réelle : le CHOIX du découpage et de la collective.
Le transport NCCL, lui, ne change pas la mathématique.

C'est le test de bout en bout du chapitre : il assemble 16.1 (column parallel), 16.2 (row
parallel), 16.3 (têtes réparties) et 16.5 (AllReduce) en un modèle complet, et il est le
garde-fou de toute implémentation distribuée future — comme 11.5 l'est pour les backends
d'attention.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
@pytest.mark.distributed
def test_tensor_parallel_model_matches_single_device_reference():
    """Roadmap 16.6 — découper un modèle ne doit RIEN changer à ses logits.

    Objectif d'apprentissage
    ------------------------
    Le parallélisme de tenseurs n'est pas une approximation : c'est une réécriture exacte du
    même calcul, à l'arrondi près. Si un modèle découpé sur 2 devices ne prédit pas les mêmes
    tokens que le même modèle sur un seul device, il y a un bug — jamais « une différence
    acceptable ». D'où l'importance de la règle d'or du découpage d'un bloc Transformer :

        q_proj, k_proj, v_proj, gate_proj, up_proj  -> COLUMN parallel (dim de sortie)
        o_proj, down_proj                           -> ROW    parallel (dim d'entrée)

    Pourquoi cet appariement précis ? Parce que la sortie d'une couche column parallel est
    DÉJÀ découpée exactement comme l'entrée attendue par la couche row parallel qui suit.
    Aucune redistribution, aucun AllGather intermédiaire : un couple column -> row ne coûte
    qu'UN SEUL AllReduce, placé après la couche row parallel. Un bloc Transformer contient deux
    tels couples — attention (qkv -> o_proj) et MLP (gate/up -> down_proj) — donc 2 AllReduce
    par bloc, et rien d'autre. Inverser le schéma (row puis column) forcerait une collective au
    milieu de chaque sous-bloc : deux fois plus de communication sur le chemin critique du
    decode, pour le même résultat.

    Schéma mental
    -------------
        input_ids (batch=1, seq=4) --embed (répliqué)--> h (1, 4, 8)

        pour chacun des 2 blocs :
            qkv column parallel -> 2 têtes par device -> attention locale (1, 2, 4, 2)
            o_proj row parallel -> 2 partiels (1, 4, 8) -> AllReduce  [collective 1]
            gate/up column parallel -> SwiGLU local (1, 4, 8)
            down_proj row parallel  -> 2 partiels (1, 4, 8) -> AllReduce  [collective 2]

        lm_head (répliqué) --> logits (1, 4, 32)

        2 blocs x 2 AllReduce = 4 collectives ; écart mesuré vs mono-device : ~4e-7

    Ce que ce test vérifie
    ----------------------
    1. le contexte d'exécution : aucun process group `torch.distributed` n'est initialisé, tout
       est simulé dans un seul processus, et le modèle expose son `tensor_parallel_size` ;
    2. le contrat de sortie est identique à celui du modèle mono-device : logits (1, 4, 32) en
       float32, tous finis ;
    3. l'équivalence numérique à la tolérance fp32 rtol=1e-5 / atol=1e-6, y compris sur le
       dernier token (celui qui sert au decode), avec la conséquence observable : les tokens
       prédits sont EXACTEMENT les mêmes ;
    4. le coût en communication est celui de la règle d'or : 2 AllReduce par bloc, soit 4 pour
       un modèle de 2 blocs — ni plus, ni moins.

    Sur la tolérance : le chemin parallèle somme les contributions partielles au lieu de faire
    une seule réduction, et l'addition flottante n'est pas associative. L'écart attendu en fp32
    est de l'ordre de 1e-7 (4e-7 mesuré sur 50 tirages) : c'est du bruit d'arrondi, on compare
    avec `torch.testing.assert_close`, jamais avec `torch.equal`. Un écart de l'ordre de 1e-2
    ne serait PAS de l'arrondi : ce serait un découpage inversé (concaténation là où il fallait
    sommer), un AllReduce oublié, ou des têtes recollées dans le mauvais ordre.

    API à faire émerger (cible roadmap : « future distributed », cible proposée
    `src/inference_lab/distributed/model.py`)
    ----------------------------------------------------------------------------
        class TensorParallelModel(torch.nn.Module):
            def __init__(self, model: torch.nn.Module, tensor_parallel_size: int) -> None: ...
            def forward(self, input_ids: torch.Tensor) -> torch.Tensor: ...

            tensor_parallel_size: int
            all_reduce_count: int   # collectives simulées du dernier forward, remis à 0 au début

    Indice : `TensorParallelModel` ne réimplémente rien, il ORCHESTRE — il découpe les poids du
    modèle reçu avec `shard_column_parallel` / `shard_row_parallel`
    (`inference_lab.distributed.tensor_parallel`) et recombine avec `all_reduce`
    (`inference_lab.distributed.collectives`). Simplification assumée de cette simulation :
    l'embedding et `lm_head` restent RÉPLIQUÉS sur tous les devices, donc non découpés ; les
    découper (vocab parallel) demanderait un AllGather de plus et n'ajouterait rien au concept.
    Deux précautions rendent le test fiable : les deux chemins doivent partir des MÊMES poids
    (donc découper le modèle de référence lui-même, pas une seconde instanciation) et le
    modèle doit être en `eval()`, le calcul sous `torch.no_grad()`. Le vrai piège de fond : le
    SwiGLU du MLP doit être appliqué AVANT l'AllReduce, sur les tranches locales de
    `gate_proj` et `up_proj`, car `silu(a) + silu(b) != silu(a + b)` — c'est précisément
    pour cela que ces deux couches sont column parallel.
    """

    pytest.skip("Roadmap TDD 16.6 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.distributed.model import TensorParallelModel

    # Arrange — dimensions jouets de la section : vocab=32, hidden=8, intermediate=16,
    #           num_heads=4, head_dim=2, num_layers=2, seq=4, batch=1,
    #           tensor_parallel_size=2, tout en `torch.float32` sur CPU.
    #           `reference_model` : un modèle mono-device jouet en `eval()`, poids
    #           déterministes (seed fixée) : embedding (32, 8), 2 blocs contenant chacun une
    #           attention causale à 4 têtes (`q_proj`, `k_proj`, `v_proj`, `o_proj`) et un MLP
    #           SwiGLU (`gate_proj`, `up_proj`, `down_proj`), puis `lm_head` (32, 8). Réutilise
    #           le `CustomLLM` de la section 11 si tu l'as déjà écrit ; sinon un petit
    #           `torch.nn.Module` local suffit, à condition de garder ces noms de sous-modules.
    #           `tp_model` : `TensorParallelModel(reference_model, tensor_parallel_size=2)`,
    #           qui découpe les poids DE `reference_model` — c'est ce qui garantit que les deux
    #           chemins partent des mêmes valeurs.
    #           `input_ids` : tenseur d'entiers (1, 4) à valeurs dans [0, 32), déterministe,
    #           STRICTEMENT le même pour les deux chemins.

    # Act — sous `torch.no_grad()`, calculer `logits_reference = reference_model(input_ids)`
    #       puis `logits_tp = tp_model(input_ids)`.

    # Assert 1 — tout se passe dans un seul processus, sans NCCL ni process group
    assert not torch.distributed.is_initialized()
    assert isinstance(tp_model, TensorParallelModel)
    assert tp_model.tensor_parallel_size == 2

    # Assert 2 — même contrat de sortie que le modèle mono-device
    assert logits_reference.shape == (1, 4, 32)
    assert logits_tp.shape == (1, 4, 32)
    assert logits_tp.dtype is torch.float32
    assert bool(logits_tp.isfinite().all())

    # Assert 3 — équivalence numérique et conséquence observable : mêmes tokens prédits
    torch.testing.assert_close(logits_tp, logits_reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        logits_tp[:, -1, :], logits_reference[:, -1, :], rtol=1e-5, atol=1e-6
    )
    assert torch.equal(logits_tp.argmax(dim=-1), logits_reference.argmax(dim=-1))

    # Assert 4 — la règle d'or : 2 AllReduce par bloc, 4 pour 2 blocs
    assert tp_model.all_reduce_count == 4
