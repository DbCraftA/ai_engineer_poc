"""Section 14.6 — allocateur de blocs : libérer, remettre au pool, réutiliser.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/14_engine/test_allocator.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(pool de 8 blocs, allocations [0, 1, 2] puis [3, 4], retour à 8 blocs libres) : c'est
volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule
que le code testé.

Cette section ferme le cycle de vie ouvert en 14.4 : les blocs de taille fixe ne servent à
rien si les blocs rendus par une séquence terminée ne reviennent pas dans le pool. C'est
l'allocateur qui donne son sens au compteur `num_free_blocks` utilisé par l'admission (14.3),
et c'est lui que le prefix caching (14.7) enrichira d'un compteur de références.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices. Même raison pour l'import
# du module cible : c'est ton code d'`Act` qui l'appellera, le linter le voit donc inutilisé.
# ruff: noqa: F401, F821

import pytest


@pytest.mark.tdd
def test_block_allocator_reuses_released_blocks():
    """Roadmap 14.6 — un bloc libéré revient au pool et resert tel quel.

    Objectif d'apprentissage
    ------------------------
    Tous les blocs d'un cache paginé ont exactement la même taille. Cette uniformité a une
    conséquence énorme : n'importe quel bloc libre convient à n'importe quelle demande, donc
    il n'existe aucune fragmentation EXTERNE. Le pool ne se « troue » jamais, contrairement à
    un allocateur de zones contiguës de tailles variables, où l'on finit par avoir assez
    d'octets libres sans avoir de plage assez longue pour les utiliser.

    L'allocateur est donc un simple ensemble d'identifiants libres, et la seule discipline à
    tenir est comptable : ce qui est rendu revient EXACTEMENT dans le pool. Un bloc oublié à
    la libération est une fuite mémoire qui réduit silencieusement la capacité du serveur
    (l'admission de 14.3 refusera des requêtes que la carte pourrait pourtant tenir), et une
    demande supérieure au pool doit échouer AVANT de servir quoi que ce soit.

    Schéma mental
    -------------
        pool de 8 blocs, l'allocateur sert toujours les plus petits identifiants libres

            départ        : libres {0..7}                   8 libres
            allocate(3)   : -> [0, 1, 2]                     5 libres
            allocate(2)   : -> [3, 4]                        3 libres
            allocate(4)   : ValueError, pool inchangé        3 libres  (pas de service partiel)
            free([0,1,2]) : 0, 1, 2 reviennent au pool       6 libres
            allocate(2)   : -> [0, 1]  <- identifiants DÉJÀ utilisés, réattribués
            free du reste : retour à l'état initial          8 libres

    Ce que ce test vérifie
    ----------------------
    1. la comptabilité d'une allocation : 8 libres au départ, [0, 1, 2] puis [3, 4] servis
       sans recouvrement, 5 blocs occupés et 3 libres à l'arrivée ;
    2. une demande plus grande que le pool disponible échoue proprement par `ValueError`, et
       ne consomme rien : le nombre de blocs libres est inchangé après l'échec ;
    3. la libération remet les blocs au pool : 6 libres après le premier `free`, et retour
       exactement à 8 quand tout est rendu ;
    4. les identifiants libérés sont réellement RÉUTILISÉS : la nouvelle allocation rend
       [0, 1], deux blocs qui appartenaient à la séquence libérée.

    API à faire émerger (cible proposée : `src/inference_lab/engine/allocator.py`)
    ----------------------------------------------------------------------------
        class BlockAllocator:
            def __init__(self, num_blocks: int) -> None: ...
            def allocate(self, num_blocks: int) -> list[int]: ...
            def free(self, block_ids: Sequence[int]) -> None: ...
            @property
            def num_free_blocks(self) -> int: ...
            @property
            def num_allocated_blocks(self) -> int: ...

        Deux décisions sont figées par ce test, et c'est le rôle d'un test de les figer :
        - une demande impossible lève `ValueError` (plutôt que de renvoyer `None`, qui
          obligerait chaque appelant à tester le retour et se propagerait en `TypeError`
          plus loin) ;
        - l'échec est ATOMIQUE : aucun bloc n'est servi, l'état du pool est intact ;
        - l'allocation est DÉTERMINISTE : les plus petits identifiants libres d'abord, ce qui
          rend les blocs attendus prévisibles dans les tests 14.6 et 14.7.

        La roadmap indique seulement « allocator » : on propose ce chemin concret dans le
        package `engine` existant.

    Indice : une liste triée d'identifiants libres, ou un `set` que l'on trie à l'allocation,
    suffit. Vérifie la disponibilité AVANT de retirer le moindre identifiant, sinon l'échec de
    l'assert 2 laissera le pool à moitié consommé. Piège de comptage : `num_allocated_blocks`
    doit se déduire du pool (`num_blocks - num_free_blocks`) et non d'un compteur séparé
    incrémenté à la main, qui finira par diverger.
    """

    pytest.skip("Roadmap TDD 14.6 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.engine.allocator import BlockAllocator

    # Arrange — `allocator`, un `BlockAllocator` sur un pool de 8 blocs, tous libres au
    #           départ. Aucun tenseur : l'allocateur ne manipule que des identifiants de
    #           blocs, la mémoire réelle étant le grand tenseur plat du cache paginé (14.5).

    # Act — jouer le cycle de vie complet et relever l'état à chaque étape : `free_at_start`,
    #       puis une allocation de 3 blocs dans `blocks_a`, une de 2 blocs dans `blocks_b`,
    #       `free_after_b`, une tentative d'allocation de 4 blocs qui doit échouer,
    #       `free_after_failure`, la libération de `blocks_a` puis `free_after_release_a`, une
    #       nouvelle allocation de 2 blocs dans `blocks_c`, et enfin la libération de tout ce
    #       qui reste avec `free_at_end`.

    # Assert 1 — la comptabilité d'une allocation
    assert free_at_start == 8
    assert blocks_a == [0, 1, 2]
    assert blocks_b == [3, 4]
    assert set(blocks_a).isdisjoint(set(blocks_b))
    assert free_after_b == 3
    assert allocator.num_allocated_blocks + allocator.num_free_blocks == 8

    # Assert 2 — une demande impossible échoue proprement et ne consomme rien
    with pytest.raises(ValueError):
        allocator.allocate(9)
    assert free_after_failure == 3

    # Assert 3 — libérer remet les blocs au pool, jusqu'à revenir à l'état initial
    assert free_after_release_a == 6
    assert free_at_end == 8
    assert free_at_end == free_at_start
    assert allocator.num_allocated_blocks == 0

    # Assert 4 — les identifiants libérés sont bien réutilisés
    assert blocks_c == [0, 1]
    assert set(blocks_c) <= set(blocks_a)
    assert set(blocks_c).isdisjoint(set(blocks_b))
