"""Section 14.4 / 14.5 — paged KV cache : blocs de taille fixe et table de correspondance.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/14_engine/test_paged_cache.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(3 blocs pour 10 tokens, 2 slots gaspillés, table [7, 2, 5], position 9 -> (5, 1)) : c'est
volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule
que le code testé.

C'est l'idée centrale de PagedAttention, empruntée à la pagination mémoire des systèmes
d'exploitation. Le KV cache contigu de la section 4 obligeait à réserver d'avance la
longueur maximale de contexte par requête ; on le remplace par des blocs de taille fixe,
alloués à la demande et repérés par une table de correspondance logique -> physique. Prix à
payer : une fragmentation INTERNE bornée (14.4). Bénéfice : plus aucune fragmentation
externe, et le partage de blocs entre requêtes qui rend possible le prefix caching (14.7).

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices. Même raison pour l'import
# du module cible : c'est ton code d'`Act` qui l'appellera, le linter le voit donc inutilisé.
# ruff: noqa: F401, F821

import pytest


@pytest.mark.tdd
def test_block_allocator_assigns_fixed_size_kv_blocks():
    """Roadmap 14.4 — des blocs de taille fixe : fragmentation interne bornée et connue.

    Objectif d'apprentissage
    ------------------------
    Dans un KV cache paginé, la mémoire n'est plus allouée « par requête » mais par BLOCS de
    `block_size` tokens, tous identiques. Une séquence de N tokens occupe donc `ceil(N /
    block_size)` blocs, et son dernier bloc n'est presque jamais plein : les slots restants
    sont réservés, comptés, mais inutilisés. C'est la fragmentation INTERNE, et elle est
    bornée par `block_size - 1` slots par séquence — c'est-à-dire connue d'avance et
    indépendante de la longueur du contexte.

    À comparer au cache contigu de la section 4 : comme on ne sait pas combien de tokens la
    requête va générer, il faut réserver `max_model_len` positions dès le prefill. Un prompt
    de 10 tokens sur un modèle à 2048 positions gaspille alors 2038 slots, soit plus de 99 %
    de la réservation. Multiplié par le coût par token calculé en 5.2 (12 KiB pour
    Qwen2.5-0.5B), c'est ce gâchis qui limitait le nombre de requêtes simultanées.

    Schéma mental
    -------------
        block_size = 4, prompt de N = 10 tokens

            bloc 0 : [t0 t1 t2 t3]     plein
            bloc 1 : [t4 t5 t6 t7]     plein
            bloc 2 : [t8 t9 .  . ]     2 tokens, 2 slots réservés et perdus

        ceil(10/4) = 3 blocs = 12 slots de capacité pour 10 tokens utiles
        fragmentation interne = 2 slots, soit au pire block_size - 1 = 3, jamais plus

        cache contigu, max_model_len = 2048 : 2048 réservés pour 10 tokens = 2038 perdus

    Ce que ce test vérifie
    ----------------------
    1. le nombre de blocs est bien `ceil(N / block_size)` : 3 blocs pour 10 tokens, 2 pour 8
       (multiple exact), 1 pour un seul token, 0 pour une séquence vide ;
    2. le dernier bloc est partiellement rempli : 12 slots de capacité, 10 utilisés,
       2 gaspillés, et 2 tokens seulement dans le dernier bloc ;
    3. la fragmentation interne est BORNÉE par `block_size - 1` : nulle sur un multiple exact
       de 4, maximale (3 slots) à 9 tokens, et strictement inférieure à 4 dans tous les cas ;
    4. l'ordre de grandeur face au cache contigu : 2 slots perdus contre 2038, ce qui rend le
       gâchis du paginé négligeable et, surtout, indépendant de `max_model_len`.

    API à faire émerger (cible proposée : `src/inference_lab/engine/paged_cache.py`)
    ------------------------------------------------------------------------------
        def num_blocks_for(num_tokens: int, block_size: int) -> int: ...

        @dataclass(frozen=True)
        class BlockLayout:
            num_blocks: int
            capacity_slots: int
            used_slots: int
            wasted_slots: int
            last_block_fill: int

        def block_layout(num_tokens: int, block_size: int) -> BlockLayout: ...
        def contiguous_wasted_slots(num_tokens: int, max_model_len: int) -> int: ...

        La roadmap indique seulement « memory manager » : on propose ce chemin concret, dans
        le package `engine` existant, partagé avec le test 14.5.

    Indice : `-(-num_tokens // block_size)` ou `math.ceil` donnent le nombre de blocs ; évite
    `int(num_tokens / block_size) + 1`, qui se trompe d'un bloc sur les multiples exacts.
    Cas limite à traiter explicitement : `num_tokens = 0` doit rendre 0 bloc, pas 1. Et
    `last_block_fill` vaut `block_size` quand la séquence remplit exactement son dernier
    bloc, jamais 0.
    """

    pytest.skip("Roadmap TDD 14.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.engine.paged_cache import (
        block_layout,
        contiguous_wasted_slots,
        num_blocks_for,
    )

    # Arrange — trois entiers seulement, aucun tenseur : `block_size` = 4 tokens par bloc,
    #           `prompt_len` = 10 tokens (volontairement NON multiple de la taille de bloc,
    #           c'est tout l'intérêt du test) et `max_model_len` = 2048, la longueur de
    #           contexte qu'un cache contigu devrait réserver d'avance pour cette requête.

    # Act — demander le nombre de blocs du prompt, puis le détail de son occupation dans
    #       `layout`, et enfin le gâchis qu'aurait produit une réservation contiguë dans
    #       `contiguous_waste`.

    # Assert 1 — le nombre de blocs est un plafond, pas une division
    assert num_blocks_for(prompt_len, block_size) == 3
    assert num_blocks_for(8, block_size) == 2
    assert num_blocks_for(1, block_size) == 1
    assert num_blocks_for(0, block_size) == 0

    # Assert 2 — le dernier bloc n'est que partiellement rempli
    assert layout.num_blocks == 3
    assert layout.capacity_slots == 12
    assert layout.used_slots == 10
    assert layout.wasted_slots == 2
    assert layout.last_block_fill == 2

    # Assert 3 — la fragmentation interne est bornée par block_size - 1
    assert layout.wasted_slots < block_size
    assert block_layout(8, block_size).wasted_slots == 0
    assert block_layout(9, block_size).wasted_slots == 3
    assert block_layout(9, block_size).last_block_fill == 1

    # Assert 4 — face au cache contigu, ce gâchis est négligeable
    assert contiguous_waste == 2038
    assert contiguous_wasted_slots(prompt_len, max_model_len) == 2038
    assert contiguous_waste > 1000 * layout.wasted_slots


@pytest.mark.tdd
def test_logical_kv_blocks_map_to_physical_blocks():
    """Roadmap 14.5 — la block table traduit un bloc logique en bloc physique quelconque.

    Objectif d'apprentissage
    ------------------------
    Une séquence voit ses tokens comme une suite continue : positions 0, 1, 2, ... En mémoire,
    ses blocs peuvent être n'importe où dans le pool physique, dans n'importe quel ordre. La
    `block table` est la table de correspondance qui fait le lien, exactement comme la table
    des pages d'un système d'exploitation :

        position -> (bloc logique = position // block_size, offset = position % block_size)
        bloc logique -> bloc physique = block_table[bloc logique]

    C'est cette indirection qui supprime la fragmentation EXTERNE : plus besoin de trouver
    2048 positions consécutives libres, n'importe quels blocs épars suffisent. C'est aussi
    elle qui permet à deux séquences de pointer vers le MÊME bloc physique, base du prefix
    caching (14.7) et du partage entre échantillons d'un même prompt. Le prix est un accès
    indirect dans le kernel d'attention, qui doit lire la table avant les clés et valeurs.

    Schéma mental
    -------------
        block_size = 4, prompt de 10 tokens, 3 blocs logiques

            block_table = [7, 2, 5]        <- blocs physiques NON contigus, non triés

            logique 0 -> physique 7   positions 0..3
            logique 1 -> physique 2   positions 4..7
            logique 2 -> physique 5   positions 8..9  (dernier bloc à moitié plein)

        position 9 : 9 // 4 = bloc logique 2, 9 % 4 = offset 1  ->  (physique 5, offset 1)
        slot plat dans le pool : 5 x 4 + 1 = 21

    Ce que ce test vérifie
    ----------------------
    1. la table a une entrée par bloc logique (3) et ses blocs physiques ne sont ni contigus
       ni triés : ils s'étalent de 2 à 7 alors que 3 blocs suffiraient à la séquence ;
    2. la traduction position -> (bloc physique, offset) sur les quatre positions
       intéressantes : 0, la dernière du premier bloc, la première du deuxième, et la
       dernière écrite ;
    3. l'index plat dans le pool physique, `bloc_physique * block_size + offset`, qui est
       l'adresse réellement utilisée par le kernel d'attention ;
    4. une position au-delà des tokens écrits lève `IndexError` : la capacité du dernier bloc
       (12 slots) ne vaut pas autorisation de lire un slot encore vide.

    API à faire émerger (cible proposée : `src/inference_lab/engine/paged_cache.py`)
    ------------------------------------------------------------------------------
        class BlockTable:
            def __init__(
                self, physical_blocks: Sequence[int], block_size: int, num_tokens: int
            ) -> None: ...
            @property
            def physical_blocks(self) -> list[int]: ...
            def slot_for_position(self, position: int) -> tuple[int, int]: ...
            def flat_slot(self, position: int) -> int: ...

        Décision documentée par ce test : `slot_for_position` lève `IndexError` pour toute
        position négative ou supérieure ou égale à `num_tokens`, plutôt que de renvoyer un
        slot réservé mais non écrit.

        La roadmap indique « paged cache » : on propose le module concret de 14.4.

    Indice : deux divisions euclidiennes, rien de plus (`divmod(position, block_size)` fait
    les deux d'un coup). Piège : ne suppose jamais que `physical_blocks` est trié ou contigu —
    tout le test est construit pour punir un code qui calculerait
    `premier_bloc + bloc_logique` au lieu de lire la table.
    """

    pytest.skip("Roadmap TDD 14.5 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.engine.paged_cache import BlockTable

    # Arrange — `block_size` = 4, `prompt_len` = 10 tokens, et `physical_blocks`, la liste des
    #           trois identifiants de blocs physiques attribués à la séquence, dans l'ordre
    #           logique : elle doit être volontairement NON contiguë et NON triée (les
    #           identifiants attendus par les asserts sont 7 puis 2 puis 5). Construire
    #           `table`, un `BlockTable` sur ces trois blocs.

    # Act — traduire quelques positions logiques en couples (bloc physique, offset), puis en
    #       index plats dans le pool.

    # Assert 1 — une entrée par bloc logique, des blocs physiques quelconques
    assert table.physical_blocks == [7, 2, 5]
    assert len(table.physical_blocks) == 3
    assert sorted(table.physical_blocks) != table.physical_blocks
    assert max(table.physical_blocks) - min(table.physical_blocks) == 5

    # Assert 2 — traduction position -> (bloc physique, offset)
    assert table.slot_for_position(0) == (7, 0)
    assert table.slot_for_position(3) == (7, 3)
    assert table.slot_for_position(4) == (2, 0)
    assert table.slot_for_position(9) == (5, 1)

    # Assert 3 — index plat dans le pool physique : bloc x block_size + offset
    assert table.flat_slot(0) == 28
    assert table.flat_slot(4) == 8
    assert table.flat_slot(9) == 21

    # Assert 4 — au-delà des tokens écrits, l'accès est une erreur
    with pytest.raises(IndexError):
        table.slot_for_position(prompt_len)
    with pytest.raises(IndexError):
        table.slot_for_position(-1)
