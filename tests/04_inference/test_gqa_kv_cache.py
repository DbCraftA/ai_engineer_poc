"""Section 4.8 — GQA : le cache stocke les têtes KV, pas les têtes de requête.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/04_inference/test_gqa_kv_cache.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(shapes (1, 1, 3, 4) contre (1, 2, 3, 4), 192 octets contre 384, ratio 2) : c'est volontaire.
Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code
testé.

Cette section combine GQA (2.8) et le KV cache (4.2) : c'est exactement pour la taille du
cache que les architectures modernes réduisent le nombre de têtes K/V. Le gain se lit dans une
seule dimension du tenseur stocké, et il se chiffre : le ratio `num_heads / num_kv_heads`.
La section 5.4 rejouera ce calcul à l'échelle d'un modèle complet.

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
def test_gqa_kv_cache_stores_only_kv_heads_not_query_heads():
    """Roadmap 4.8 — avec GQA, la mémoire du cache est divisée par num_heads / num_kv_heads.

    Objectif d'apprentissage
    ------------------------
    En GQA, plusieurs têtes de requête partagent une même tête clé/valeur. Les têtes de
    requête ne sont jamais stockées — Q est consommé immédiatement puis jeté — donc le seul
    axe « têtes » du cache est `num_kv_heads`. Confondre les deux est l'erreur qui fait
    exploser la mémoire d'un moteur maison, ou qui produit un cache trois fois trop gros
    sans que rien n'échoue fonctionnellement.

    Le gain est exactement proportionnel : Qwen2.5-0.5B a 14 têtes de requête pour 2 têtes
    KV, soit un cache 7 fois plus léger qu'en MHA, à qualité quasi inchangée. Ici, avec 2
    têtes de requête et 1 tête KV, le facteur est 2.

    Schéma mental
    -------------
        batch=1, num_layers=2, num_heads=2, num_kv_heads=1, head_dim=4, 3 tokens, float32

        MHA (num_kv_heads = num_heads = 2)     GQA (num_kv_heads = 1)
          K (1, 2, 3, 4) = 24 éléments           K (1, 1, 3, 4) = 12 éléments
          V (1, 2, 3, 4) = 24 éléments           V (1, 1, 3, 4) = 12 éléments
          x 2 couches x 4 octets = 384 o         x 2 couches x 4 octets = 192 o
          128 octets par token                   64 octets par token

        côté attention : les 2 têtes de requête lisent la MÊME tête KV
          K (1, 1, 3, 4) --repeat_interleave(2, dim=1)--> (1, 2, 3, 4)

    Ce que ce test vérifie
    ----------------------
    1. le cache GQA stocke `num_kv_heads` = 1 tête par couche, shape (1, 1, 3, 4), et non
       les 2 têtes de requête ;
    2. la mémoire chiffrée : 192 octets en GQA contre 384 octets en MHA sur le même prompt,
       soit 64 contre 128 octets par token ;
    3. le rapport des deux vaut exactement `num_heads / num_kv_heads` = 2 ;
    4. le partage est bien un partage : en répliquant la tête KV pour ses 2 têtes de
       requête, on retrouve (1, 2, 3, 4) avec deux têtes strictement identiques.

    API à faire émerger (cible roadmap « cache », cible proposée :
    `src/inference_lab/cache/kv_cache.py`, module déjà visé par 4.2 / 4.3)
    ---------------------------------------------------------------------
        class KVCache:
            def __init__(self, num_layers: int) -> None: ...

        def kv_cache_bytes(cache: KVCache) -> int: ...

    Indice : `kv_cache_bytes` somme `numel() * element_size()` sur les deux tenseurs de
    chaque couche (formule de la section 1.9) ; la réplication des têtes KV se fait avec
    `torch.repeat_interleave(keys, 2, dim=1)`. Piège : `torch.repeat` ou `expand` sans
    `contiguous` n'ordonne pas les têtes de la même façon — vérifie que la tête de requête 0
    et la tête 1 pointent bien sur la même tête KV.
    """

    pytest.skip("Roadmap TDD 4.8 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.cache.kv_cache import KVCache, kv_cache_bytes

    # Arrange — dimensions jouets constantes : batch=1, num_layers=2, num_heads=2,
    #           num_kv_heads=1, head_dim=4, prompt de 3 tokens, float32. Construire
    #           `gqa_cache`, un KVCache de 2 couches rempli avec des paires (K, V) de shape
    #           (1, 1, 3, 4), et `mha_cache`, le même cache de comparaison rempli avec des
    #           paires de shape (1, 2, 3, 4) — une tête KV par tête de requête.

    # Act — construire `expanded_keys` en répliquant la tête KV de la couche 0 du cache GQA
    #       pour les 2 têtes de requête (axe 1).

    # Assert 1 — le cache stocke les têtes KV, pas les têtes de requête
    assert isinstance(gqa_cache, KVCache)
    assert gqa_cache.get(0)[0].shape == (1, 1, 3, 4)
    assert gqa_cache.get(0)[0].shape[1] == 1
    assert mha_cache.get(0)[0].shape == (1, 2, 3, 4)

    # Assert 2 — mémoire chiffrée : 192 octets contre 384, soit 64 contre 128 par token
    assert kv_cache_bytes(gqa_cache) == 192
    assert kv_cache_bytes(mha_cache) == 384
    assert kv_cache_bytes(gqa_cache) // 3 == 64

    # Assert 3 — le gain vaut exactement num_heads / num_kv_heads
    assert kv_cache_bytes(mha_cache) // kv_cache_bytes(gqa_cache) == 2

    # Assert 4 — une tête KV partagée par 2 têtes de requête
    assert expanded_keys.shape == (1, 2, 3, 4)
    assert torch.equal(expanded_keys[:, 0], expanded_keys[:, 1])
