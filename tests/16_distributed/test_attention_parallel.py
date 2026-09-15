"""Section 16.3 — répartir les têtes d'attention entre devices simulés.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/16_distributed/test_attention_parallel.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(2 têtes par device, shard (1, 2, 4, 2), reconstitution (1, 4, 4, 2)) : c'est volontaire. Un
test doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Dimensions jouets constantes de toute la section 16 : hidden=8, intermediate=16, num_heads=4,
head_dim=2, seq=4, batch=1, tensor_parallel_size=2 — soit 4 têtes / 2 devices = 2 têtes chacun.

Pourquoi ce test tourne sur UN SEUL processus
--------------------------------------------
Le marker `distributed` est conservé, mais il n'y a ici ni
`torch.distributed.init_process_group`, ni NCCL, ni second GPU. On SIMULE le parallélisme en
découpant les poids en shards (un par « device ») et en recombinant les résultats à la main.
Ce que l'on veut prouver est purement mathématique — quel découpage, quelle collective — et
cela n'a besoin d'aucune communication réelle pour être vérifié.

Ce test est la lecture « attention » du test 16.1 : découper `q_proj`, `k_proj` et `v_proj` en
column parallel revient exactement à distribuer les TÊTES, parce que la dimension de sortie de
ces projections EST la concaténation des têtes. C'est ce qui rend le parallélisme de tenseurs
naturel dans un Transformer : les têtes sont déjà indépendantes les unes des autres.

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
def test_attention_heads_can_be_partitioned_across_devices():
    """Roadmap 16.3 — une tête d'attention est l'unité indivisible du découpage.

    Objectif d'apprentissage
    ------------------------
    Le parallélisme de tenseurs ne coupe jamais une tête en deux : il distribue des têtes
    ENTIÈRES. La raison est structurelle — le softmax d'une tête porte sur ses propres scores,
    donc couper `head_dim` obligerait à communiquer AU MILIEU du calcul d'attention, alors que
    couper au niveau des têtes ne demande aucune communication du tout jusqu'à `o_proj`. Chaque
    device fait une attention complète, mais sur 2 têtes seulement au lieu de 4. Sur
    Qwen2.5-0.5B (14 têtes Q, 2 têtes KV en GQA), c'est la contrainte qui décide des valeurs de
    `tensor_parallel_size` admissibles : 2 et 7 pour les têtes Q, mais 2 seulement si l'on veut
    aussi découper les têtes KV. Au-delà, les têtes KV doivent être RÉPLIQUÉES sur les devices,
    ce qui duplique le KV cache — le prix à payer, à connaître avant de choisir sa topologie.

    Schéma mental
    -------------
        Wq (hidden=8, hidden=8) --split(dim=0, column parallel)--> Wq0 (4, 8)  Wq1 (4, 8)

        device 0 : x (1, 4, 8) --Wq0--> (1, 4, 4) --view/transpose--> (1, 2, 4, 2)  têtes 0, 1
        device 1 : x (1, 4, 8) --Wq1--> (1, 4, 4) --view/transpose--> (1, 2, 4, 2)  têtes 2, 3

        cat(dim=1) -> (batch=1, num_heads=4, seq=4, head_dim=2) = référence mono-device

        GQA : num_kv_heads=2, 2 % 2 == 0 -> 1 tête KV par device, OK
              num_kv_heads=3, 3 % 2 == 1 -> impossible : répliquer les têtes KV partout

    Ce que ce test vérifie
    ----------------------
    1. la répartition : 4 têtes / 2 devices = 2 têtes par device, un shard de têtes a la shape
       (batch=1, heads_locales=2, seq=4, head_dim=2) et le shard de poids column parallel
       correspondant la shape (4, 8) ;
    2. quelles têtes vont où : le device 0 reçoit exactement les têtes 0 et 1, le device 1 les
       têtes 2 et 3, valeur par valeur ;
    3. la reconstitution : concaténer les shards sur la dimension des têtes rend la référence —
       à l'identique quand on repart d'un simple découpage, à la tolérance fp32
       rtol=1e-5 / atol=1e-6 quand on passe par les shards de poids column parallel ;
    4. la contrainte de divisibilité, y compris pour les têtes KV en GQA : un nombre de têtes
       non divisible par le nombre de devices est refusé.

    API à faire émerger (cible roadmap : « future distributed », cible proposée
    `src/inference_lab/distributed/tensor_parallel.py`)
    ----------------------------------------------------------------------------
        def heads_per_device(num_heads: int, tensor_parallel_size: int) -> int: ...

        def shard_attention_heads(
            heads: torch.Tensor, tensor_parallel_size: int
        ) -> list[torch.Tensor]: ...

        def merge_attention_heads(shards: list[torch.Tensor]) -> torch.Tensor: ...

    Indice : sur un tenseur au layout (batch, num_heads, seq, head_dim) — celui de la section
    2.7 — tout se joue sur la dimension 1 : `heads.split(heads_per_device(...), dim=1)` pour
    découper, `torch.cat(shards, dim=1)` pour recoller. Piège : ne découpe pas la dimension
    `head_dim` (la dernière), ce qui donnerait des shapes plausibles mais des demi-têtes
    mathématiquement fausses — l'assert 2 le détecte. Lève une `ValueError` si
    `num_heads % tensor_parallel_size != 0` : c'est la même règle pour les têtes Q et pour les
    têtes KV, et c'est elle qui impose la réplication du KV cache en GQA.
    """

    pytest.skip("Roadmap TDD 16.3 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.distributed.tensor_parallel import (
        heads_per_device,
        merge_attention_heads,
        shard_attention_heads,
        shard_column_parallel,
    )

    # Arrange — dimensions jouets de la section : hidden=8, num_heads=4, head_dim=2, seq=4,
    #           batch=1, tensor_parallel_size=2, tout en `torch.float32` sur CPU.
    #           `x` : tenseur (1, 4, 8) déterministe (seed fixée), vu en entier par chaque
    #           device simulé.
    #           `w_q` : poids de `q_proj` sans biais, shape (8, 8) au format `torch.nn.Linear`.
    #           `heads_reference` : les têtes de la référence mono-device, obtenues en
    #           projetant `x` par `w_q` puis en relisant le résultat au layout
    #           (batch, num_heads, seq, head_dim) = (1, 4, 4, 2) comme en section 2.7.
    #           `head_shards` : la liste des têtes LOCALES, obtenue en projetant `x` par chaque
    #           shard de `shard_column_parallel(w_q, 2)` puis en relisant chaque résultat au
    #           layout (1, 2, 4, 2) — 2 têtes par device.

    # Act — appeler `heads_per_device(4, 2)`, `shard_attention_heads(heads_reference, 2)` et
    #       `merge_attention_heads` sur les deux listes de shards.

    # Assert 1 — 4 têtes réparties sur 2 devices : 2 têtes chacun
    assert heads_per_device(4, 2) == 2
    assert len(head_shards) == 2
    assert head_shards[0].shape == (1, 2, 4, 2)
    assert heads_reference.shape == (1, 4, 4, 2)
    assert shard_column_parallel(w_q, 2)[0].shape == (4, 8)

    # Assert 2 — quelles têtes vont sur quel device, valeur par valeur
    assert torch.equal(shard_attention_heads(heads_reference, 2)[0], heads_reference[:, 0:2])
    assert torch.equal(shard_attention_heads(heads_reference, 2)[1], heads_reference[:, 2:4])

    # Assert 3 — la reconstitution recolle les têtes dans l'ordre des rangs
    assert torch.equal(
        merge_attention_heads(shard_attention_heads(heads_reference, 2)), heads_reference
    )
    assert merge_attention_heads(head_shards).shape == (1, 4, 4, 2)
    torch.testing.assert_close(
        merge_attention_heads(head_shards), heads_reference, rtol=1e-5, atol=1e-6
    )

    # Assert 4 — divisibilité : la règle vaut aussi pour les têtes KV de GQA
    assert heads_per_device(2, 2) == 1
    with pytest.raises(ValueError):
        heads_per_device(3, 2)
    with pytest.raises(ValueError):
        shard_attention_heads(heads_reference, 3)
