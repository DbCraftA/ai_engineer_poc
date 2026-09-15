"""Section 14.7 — prefix caching : partager les blocs d'un préfixe commun.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/14_engine/test_prefix_cache.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(tables [0, 1, 2, 3] et [0, 1, 4, 5], 6 blocs au lieu de 8, 8 tokens de prefill évités) :
c'est volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même
formule que le code testé.

C'est le bénéfice que l'indirection de 14.5 rendait possible : si deux requêtes commencent
par les mêmes tokens, leurs premiers blocs contiennent exactement les mêmes clés et valeurs
(l'attention est causale, un token ne dépend que de ses prédécesseurs). Deux block tables
peuvent donc pointer vers les MÊMES blocs physiques, ce qui économise à la fois de la
mémoire et du calcul de prefill. C'est ce qui rend les longs prompts système quasi gratuits
à partir de la deuxième requête. Contrepartie : il faut compter les références pour ne pas
libérer un bloc encore utilisé.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices. Même raison pour l'import
# du module cible : c'est ton code d'`Act` qui l'appellera, le linter le voit donc inutilisé.
# ruff: noqa: F401, F821

import pytest


@pytest.mark.tdd
def test_identical_prefix_can_reuse_cached_kv_blocks():
    """Roadmap 14.7 — un préfixe partagé, les mêmes blocs physiques, un prefill en moins.

    Objectif d'apprentissage
    ------------------------
    Deux requêtes qui commencent par le même prompt système produisent, sur ces tokens, des
    clés et des valeurs strictement identiques : l'attention est causale, donc le KV d'une
    position ne dépend que d'elle et de ce qui précède, jamais de ce qui suit. Recalculer ce
    préfixe et le stocker deux fois est donc un pur gaspillage — de calcul au prefill, et de
    blocs dans le pool.

    Le prefix caching indexe les blocs PLEINS par le contenu du préfixe qui y mène (en
    pratique un hachage cumulé des tokens depuis la position 0) et réutilise le bloc physique
    quand la clé est déjà connue. Deux conséquences pratiques : le TTFT d'une requête à long
    prompt système s'écroule, et un bloc partagé ne peut plus être libéré par la première
    séquence qui s'en va — d'où le compteur de références, exactement comme un `Rc` ou un
    `shared_ptr`. Attention à la granularité : seuls les blocs COMPLETS sont partageables, un
    bloc partiellement rempli n'est pas encore figé.

    Schéma mental
    -------------
        block_size = 4, pool de 8 blocs
        prompt système commun : 8 tokens = exactement 2 blocs pleins

            requête A : 8 tokens communs + 6 tokens propres = 14 -> ceil(14/4) = 4 blocs
                        block_table_a = [0, 1, 2, 3]        4 blocs neufs
            requête B : 8 tokens communs + 6 tokens propres = 14 -> 4 blocs logiques
                        block_table_b = [0, 1, 4, 5]        2 réutilisés, 2 neufs seulement

        blocs physiques réellement consommés : 6 au lieu de 8   (-25 %)
        prefill évité pour B : 8 tokens sur 14, il n'en reste que 6 à calculer
        refcount des blocs 0 et 1 : 2 -> libérer A ne les rend PAS au pool

    Ce que ce test vérifie
    ----------------------
    1. la première requête ne réutilise rien : 4 blocs neufs [0, 1, 2, 3], 0 bloc réutilisé,
       0 token mis en cache à son profit, et 4 blocs encore libres sur 8 ;
    2. la seconde requête réutilise les MÊMES blocs physiques pour le préfixe et diverge
       ensuite : sa table vaut [0, 1, 4, 5], son préfixe est identique à celui de A et son
       suffixe est disjoint ;
    3. le gain chiffré : 2 blocs neufs au lieu de 4, 6 blocs consommés au lieu de 8, et 8 des
       14 tokens de prompt dont le prefill n'est pas refait ;
    4. le comptage de références protège le partage : libérer A ne rend que ses 2 blocs
       propres (4 libres, pas 6) et les blocs du préfixe restent vivants pour B.

    API à faire émerger (cible proposée : `src/inference_lab/engine/prefix_cache.py`)
    -------------------------------------------------------------------------------
        @dataclass(frozen=True)
        class PrefixAllocation:
            block_table: list[int]
            num_reused_blocks: int
            num_cached_tokens: int          # num_reused_blocks x block_size

        class PrefixCachingAllocator:
            def __init__(self, num_blocks: int, block_size: int) -> None: ...
            def allocate_for_prompt(self, token_ids: Sequence[int]) -> PrefixAllocation: ...
            def release(self, block_table: Sequence[int]) -> None: ...
            def ref_count(self, block_id: int) -> int: ...
            @property
            def num_free_blocks(self) -> int: ...

        La roadmap indique seulement « prefix cache » : on propose ce chemin concret dans le
        package `engine` existant. L'allocateur nu de 14.6 (`BlockAllocator`) est la
        dépendance naturelle : ce module lui ajoute l'index de préfixes et les références.

    Indice : itère sur le prompt par tranches de `block_size` en maintenant la clé cumulée
    (`tuple` des tokens depuis 0, ou un hachage) ; si la clé est déjà dans l'index ET que la
    tranche est pleine, incrémente le compteur de références du bloc au lieu d'en allouer un
    nouveau. Deux pièges qui font passer le test à côté du concept : indexer un bloc
    incomplet (il serait réutilisé avec un contenu qui va encore changer), et indexer une
    tranche par son seul contenu local au lieu du préfixe entier (deux blocs identiques mais
    précédés de contextes différents contiennent des KV différents).
    """

    pytest.skip("Roadmap TDD 14.7 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.engine.prefix_cache import PrefixCachingAllocator

    # Arrange — `block_size` = 4 et `allocator`, un `PrefixCachingAllocator` sur un pool de
    #           8 blocs. Trois listes d'identifiants de tokens : `system_prompt` de 8 tokens
    #           (soit exactement 2 blocs PLEINS, condition nécessaire au partage),
    #           `question_a` et `question_b` de 6 tokens chacune, dont les contenus doivent
    #           différer dès leur premier token. Les deux prompts complets sont
    #           `system_prompt + question_a` et `system_prompt + question_b`, longs de
    #           14 tokens chacun.

    # Act — allouer les blocs du prompt de A dans `allocation_a` et relever `free_after_a`,
    #       puis ceux du prompt de B dans `allocation_b` et relever `free_after_b` ; libérer
    #       enfin la table de A et relever `free_after_release_a`.

    # Assert 1 — la première requête paie le prix plein : 4 blocs neufs
    assert allocation_a.block_table == [0, 1, 2, 3]
    assert allocation_a.num_reused_blocks == 0
    assert allocation_a.num_cached_tokens == 0
    assert free_after_a == 4

    # Assert 2 — la seconde partage les blocs du préfixe et diverge ensuite
    assert allocation_b.block_table == [0, 1, 4, 5]
    assert allocation_b.block_table[:2] == allocation_a.block_table[:2]
    assert set(allocation_b.block_table[2:]).isdisjoint(set(allocation_a.block_table[2:]))
    assert allocation_b.num_reused_blocks == 2

    # Assert 3 — le gain, en blocs et en tokens de prefill
    assert len(allocation_b.block_table) - allocation_b.num_reused_blocks == 2
    assert free_after_b == 2
    assert allocation_b.num_cached_tokens == 8
    assert 14 - allocation_b.num_cached_tokens == 6

    # Assert 4 — le comptage de références empêche de libérer un bloc encore partagé
    assert free_after_release_a == 4
    assert allocator.ref_count(0) == 1
    assert allocator.ref_count(1) == 1
    assert allocator.ref_count(2) == 0
