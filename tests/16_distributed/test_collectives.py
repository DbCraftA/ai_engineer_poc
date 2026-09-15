"""Section 16.4 / 16.5 — collectives AllGather et AllReduce comme fonctions pures.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/16_distributed/test_collectives.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
((1, 4, 8) après AllGather, shape inchangée après AllReduce, 1 + 2 = 3) : c'est volontaire. Un
test doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Dimensions jouets constantes de toute la section 16 : hidden=8, intermediate=16, num_heads=4,
head_dim=2, seq=4, batch=1, tensor_parallel_size=2.

Pourquoi ces tests tournent sur UN SEUL processus
-------------------------------------------------
Le marker `distributed` est conservé, mais on n'appelle jamais
`torch.distributed.init_process_group`, et il n'y a ni NCCL ni second GPU. On SIMULE les
devices : une collective devient une FONCTION PURE qui reçoit la liste des shards — un par
« device » simulé, dans l'ordre des rangs — et renvoie le résultat recombiné. Cette
simplification est le cœur pédagogique du chapitre : elle isole complètement le contenu
mathématique de la collective (concaténer ? sommer ? sur quelle dimension ? dans quel ordre ?)
de la plomberie de communication, qui ne s'apprend pas dans un test unitaire. Sur un vrai
cluster, chaque rang appelle la collective avec son seul shard et reçoit le même résultat que
tous les autres ; ici, le fait que la fonction soit pure rend cette propriété
« tous les rangs voient la même chose » vraie par construction.

Les deux collectives de ce fichier suffisent au parallélisme de tenseurs : AllGather recolle
des tranches (sortie d'une couche column parallel, section 16.1), AllReduce termine une somme
partielle (sortie d'une couche row parallel, section 16.2).

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
def test_all_gather_reconstructs_all_shards():
    """Roadmap 16.4 — AllGather : chaque rang repart avec le tenseur complet, dans l'ordre.

    Objectif d'apprentissage
    ------------------------
    AllGather répond à la question « je n'ai qu'une tranche, j'ai besoin du tout ». Chaque rang
    envoie son shard et reçoit la CONCATÉNATION de tous les shards, rangés dans l'ordre des
    rangs : shard du rang 0, puis rang 1, etc. Deux conséquences pratiques à retenir. D'abord
    l'ordre est une convention, pas un détail : si le rang 1 se croit en position 0, le tenseur
    reconstitué mélange les colonnes et le modèle produit du bruit — un bug silencieux, aucune
    erreur de shape. Ensuite le volume communiqué est proportionnel à la taille TOTALE du
    tenseur, ce qui explique pourquoi on évite les AllGather inutiles : dans un bloc
    Transformer bien découpé, la sortie column parallel alimente directement la couche row
    parallel suivante, et aucun AllGather n'est nécessaire.

    Schéma mental
    -------------
        rang 0 : shard (batch=1, seq=4, 4)      rang 1 : shard (batch=1, seq=4, 4)
                        |                                      |
                        +----------- all_gather ---------------+
                        |                                      |
        rang 0 reçoit (1, 4, 8)                 rang 1 reçoit (1, 4, 8)   <- le MÊME tenseur

        colonnes 0..3 = shard du rang 0     colonnes 4..7 = shard du rang 1
        4 + 4 = 8 : la dimension de concaténation grandit, le nombre d'éléments est conservé

    Ce que ce test vérifie
    ----------------------
    1. la shape finale : 2 shards (1, 4, 4) donnent (1, 4, 8), et aucun élément n'est perdu
       (32 éléments au total) ;
    2. l'ordre exact des shards : le rang 0 occupe les 4 premières colonnes, le rang 1 les 4
       suivantes, valeur par valeur (égalité bit à bit : une concaténation ne calcule rien) ;
    3. l'ordre fait partie du contrat : inverser les shards change le résultat ;
    4. tous les rangs obtiennent le même tenseur, et des shards de shapes incompatibles sont
       refusés.

    API à faire émerger (cible roadmap : « future distributed », cible proposée
    `src/inference_lab/distributed/collectives.py`)
    ----------------------------------------------------------------------------
        def all_gather(shards: list[torch.Tensor], dim: int = -1) -> torch.Tensor: ...

    Indice : `torch.cat(shards, dim=dim)` fait tout le travail — l'intérêt du module est de
    nommer la collective, de figer la convention « ordre = ordre des rangs » et de valider les
    entrées. Vérifie que tous les shards ont la même shape et lève une `ValueError` sinon :
    c'est exactement ce que le vrai `torch.distributed.all_gather` exige. Comme
    `torch.cat` renvoie une copie, le résultat ne partage rien avec les shards d'entrée.
    """

    pytest.skip("Roadmap TDD 16.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.distributed.collectives import all_gather

    # Arrange — dimensions jouets de la section : hidden=8, seq=4, batch=1,
    #           tensor_parallel_size=2, tout en `torch.float32` sur CPU.
    #           `shards` : liste de 2 tenseurs (1, 4, 4) déterministes (seed fixée), un par
    #           device simulé, dans l'ordre des rangs. Leurs valeurs doivent être TOUTES
    #           DISTINCTES d'un shard à l'autre, sinon l'assert 3 ne prouve rien.
    #           `mismatched` : une liste de 2 shards de shapes DIFFÉRENTES, par exemple
    #           (1, 4, 4) et (1, 4, 2), pour vérifier la validation.

    # Act — calculer `gathered = all_gather(shards)`, puis `gathered_per_rank`, la liste des
    #       résultats obtenus en appelant `all_gather(shards)` une fois par rang simulé (2 fois) :
    #       la collective étant une fonction pure, chaque rang doit voir la même chose.

    # Assert 1 — la dimension concaténée grandit, rien n'est perdu
    assert shards[0].shape == (1, 4, 4)
    assert gathered.shape == (1, 4, 8)
    assert gathered.numel() == 32
    assert gathered.dtype is torch.float32

    # Assert 2 — ordre exact : rang 0 d'abord, rang 1 ensuite, bit à bit
    assert torch.equal(gathered[..., 0:4], shards[0])
    assert torch.equal(gathered[..., 4:8], shards[1])

    # Assert 3 — l'ordre des rangs fait partie du contrat
    assert not torch.equal(gathered, all_gather(shards[::-1]))

    # Assert 4 — tous les rangs voient le même tenseur ; shards incompatibles refusés
    assert torch.equal(gathered_per_rank[0], gathered_per_rank[1])
    assert torch.equal(gathered_per_rank[0], gathered)
    with pytest.raises(ValueError):
        all_gather(mismatched)


@pytest.mark.tdd
@pytest.mark.distributed
def test_all_reduce_matches_single_device_reference_sum():
    """Roadmap 16.5 — AllReduce : terminer une somme partielle sans changer la shape.

    Objectif d'apprentissage
    ------------------------
    AllReduce répond à une autre question : « nous avons chacun une partie de la somme,
    donnez-nous le total ». Chaque rang apporte un tenseur de la MÊME shape que le résultat, et
    reçoit la somme élément par élément de toutes les contributions. C'est la collective du
    parallélisme de tenseurs : elle clôt chaque couche row parallel (`o_proj`, `down_proj`,
    section 16.2). Le contraste avec AllGather est ce qu'il faut retenir :

        AllGather : shapes partielles -> concaténation -> la shape GRANDIT
        AllReduce : shapes complètes  -> somme         -> la shape est INCHANGÉE

    C'est aussi la collective la plus coûteuse d'un moteur d'inférence distribué : elle est sur
    le chemin critique de chaque token généré, deux fois par couche. À `tensor_parallel_size`
    élevé, c'est elle qui finit par dominer la latence de decode, et c'est pourquoi on cherche
    à en avoir le moins possible plutôt qu'à les rendre plus rapides.

    Schéma mental
    -------------
        rang 0 : contribution (1, 4, 8)     rang 1 : contribution (1, 4, 8)
                        |                                  |
                        +--------- all_reduce (somme) -----+
                        |                                  |
        rang 0 reçoit (1, 4, 8)             rang 1 reçoit (1, 4, 8)   <- le MÊME tenseur

        cas trivial pour fixer les idées :  1 partout  +  2 partout  =  3 partout
        et une contribution seule NE VAUT PAS le total : elle est incomplète

    Ce que ce test vérifie
    ----------------------
    1. la shape est inchangée par la collective (1, 4, 8), contrairement à AllGather ;
    2. le résultat vaut la référence mono-device (la somme des contributions) à la tolérance
       fp32 rtol=1e-5 / atol=1e-6, alors qu'une contribution seule en diffère largement ;
    3. une valeur attendue écrite en dur sur un cas trivial : 1 + 2 = 3 partout ;
    4. tous les rangs reçoivent exactement le même tenseur, et des contributions de shapes
       incompatibles sont refusées.

    Sur la tolérance : ici les contributions sont sommées dans l'ordre des rangs, donc le
    résultat est stable et se compare même bit à bit à `torch.stack(...).sum(dim=0)`. Sur un
    vrai cluster, en revanche, NCCL utilise un arbre de réduction dont l'ordre dépend de la
    topologie et du nombre de rangs, et l'addition flottante n'est pas associative : on compare
    donc toujours avec `torch.testing.assert_close` et une tolérance, jamais avec
    `torch.equal`. Prendre cette habitude ici évite un test qui passe en simulation et échoue
    sur la vraie machine.

    API à faire émerger (cible roadmap : « future distributed », cible proposée
    `src/inference_lab/distributed/collectives.py`)
    ----------------------------------------------------------------------------
        def all_reduce(contributions: list[torch.Tensor]) -> torch.Tensor: ...

    Indice : `torch.stack(contributions, dim=0).sum(dim=0)`, ou une simple boucle d'additions.
    Valide que toutes les contributions ont la même shape et lève une `ValueError` sinon — une
    collective ne redimensionne rien. Piège de fond à ne pas reproduire côté modèle : ne somme
    JAMAIS après une non-linéarité, car `silu(a) + silu(b) != silu(a + b)` ; l'AllReduce se
    place après la projection linéaire, pas au milieu du MLP.
    """

    pytest.skip("Roadmap TDD 16.5 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.distributed.collectives import all_reduce

    # Arrange — dimensions jouets de la section : hidden=8, seq=4, batch=1,
    #           tensor_parallel_size=2, tout en `torch.float32` sur CPU.
    #           `contributions` : liste de 2 tenseurs (1, 4, 8) déterministes (seed fixée), un
    #           par device simulé — les sorties partielles d'une couche row parallel. Leurs
    #           valeurs doivent être de magnitude comparable à 1 et non nulles, pour que
    #           l'assert 2 montre bien qu'une contribution seule est fausse.
    #           `reference` : la somme mono-device des contributions, calculée avec
    #           `torch.stack(contributions, dim=0).sum(dim=0)` — l'oracle du test.
    #           `mismatched` : une liste de 2 contributions de shapes DIFFÉRENTES, par exemple
    #           (1, 4, 8) et (1, 4, 4).

    # Act — calculer `reduced = all_reduce(contributions)`, puis `reduced_per_rank`, la liste
    #       des résultats obtenus en appelant `all_reduce(contributions)` une fois par rang
    #       simulé (2 fois).

    # Assert 1 — AllReduce ne change pas la shape, contrairement à AllGather
    assert contributions[0].shape == (1, 4, 8)
    assert reduced.shape == (1, 4, 8)
    assert reduced.dtype is torch.float32

    # Assert 2 — le total vaut la référence ; une contribution seule est incomplète
    torch.testing.assert_close(reduced, reference, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(reduced, contributions[0], rtol=1e-3, atol=1e-3)

    # Assert 3 — valeur attendue en dur sur un cas trivial : 1 + 2 = 3
    assert torch.equal(
        all_reduce([torch.ones(2, 3), torch.full((2, 3), 2.0)]),
        torch.full((2, 3), 3.0),
    )

    # Assert 4 — tous les rangs voient le même total ; shapes incompatibles refusées
    assert torch.equal(reduced_per_rank[0], reduced_per_rank[1])
    assert torch.equal(reduced_per_rank[0], reduced)
    with pytest.raises(ValueError):
        all_reduce(mismatched)
