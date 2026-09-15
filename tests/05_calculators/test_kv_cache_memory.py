"""Section 5.2 / 5.3 / 5.4 — mémoire du KV cache : formule, contexte, GQA.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/05_calculators/test_kv_cache_memory.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(12 288 octets par token, 12 582 912 octets pour 1024 tokens, ratio 7 entre MHA et GQA) :
c'est volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même
formule que le code testé.

C'est LA formule de la section, celle qui découle directement de la shape posée en 4.2
(`(batch, num_kv_heads, seq, head_dim)` par couche, pour K et pour V) :

    octets_cache = 2 x num_layers x num_kv_heads x head_dim x num_tokens x octets_par_élément

Le facteur 2 compte K et V. Les trois tests ci-dessous font varier un seul terme à la fois :
le nombre de tokens (5.3, croissance linéaire) puis `num_kv_heads` (5.4, effet de GQA).

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
def test_kv_cache_memory_matches_layers_heads_tokens_formula():
    """Roadmap 5.2 — la mémoire du cache se déduit de la shape (couches, têtes KV, tokens).

    Objectif d'apprentissage
    ------------------------
    Le cache de la section 4 stocke, pour CHAQUE couche, un tenseur K et un tenseur V de
    shape (batch, num_kv_heads, seq, head_dim). Compter ses octets, c'est donc multiplier
    tous ces axes puis appliquer le dtype :

        2 (K et V) x num_layers x num_kv_heads x head_dim x num_tokens x octets_par_élément

    Cette formule est l'outil de dimensionnement d'un serveur d'inférence : elle donne le
    coût mémoire d'UNE requête, donc le nombre de requêtes simultanées qu'une carte peut
    tenir une fois les poids chargés. Noter ce qui n'y figure PAS : `num_heads` (les têtes
    de requête ne sont jamais mises en cache), `hidden` et `intermediate` (les activations
    du MLP sont jetées après chaque token).

    Schéma mental
    -------------
        Qwen2.5-0.5B : num_layers=24, num_kv_heads=2, head_dim=64, FP16 (2 octets), batch=1

        par token : 2 x 24 x 2 x 64 x 2      =     12 288 octets  = 12 KiB
        1024 tokens : 12 288 x 1024          = 12 582 912 octets  = 12 MiB
        les mêmes 1024 tokens en FP32        = 25 165 824 octets  = 24 MiB

        à comparer au 1 Go de poids FP16 de la section 5.1 : sur un contexte court le cache
        est négligeable, sur 32k tokens il pèse 384 MiB — d'où la pression de la section 5.4

    Ce que ce test vérifie
    ----------------------
    1. le coût d'un seul token, 12 288 octets, qui est le pas de croissance du cache ;
    2. le coût du contexte complet de 1024 tokens, 12 582 912 octets, soit exactement
       12 MiB ;
    3. la dépendance au dtype : le même cache en FP32 coûte exactement le double ;
    4. la formule ignore `num_heads` : passer de 14 à 28 têtes de requête, à `num_kv_heads`
       constant, ne change pas un octet du cache.

    API à faire émerger (cible roadmap : `src/inference_lab/calculators/kv_cache_memory.py`)
    ---------------------------------------------------------------------------------------
        def kv_cache_bytes(
            num_layers: int,
            num_kv_heads: int,
            head_dim: int,
            num_tokens: int,
            dtype: torch.dtype,
            batch_size: int = 1,
        ) -> int: ...

    Indice : le seul piège est le facteur 2 de K et V, oublié une fois sur deux — il donne
    un résultat deux fois trop petit, ce qui passe inaperçu jusqu'au premier OOM. Signature
    à mots-clés obligatoires si tu veux : six entiers positionnels dans le même ordre que la
    formule, c'est déjà une source d'erreur.
    """

    pytest.skip("Roadmap TDD 5.2 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.calculators.kv_cache_memory import kv_cache_bytes

    # Arrange — la configuration de référence de Qwen2.5-0.5B, en entiers : `num_layers` = 24,
    #           `num_kv_heads` = 2, `head_dim` = 64, `batch_size` = 1, et deux longueurs de
    #           contexte `single_token` = 1 et `context_tokens` = 1024. Aucun tenseur à
    #           allouer : la section 5 ne calcule que des formules fermées.

    # Act — demander à l'API les octets du cache pour 1 token puis pour 1024 tokens en
    #       `torch.float16`, et stocker les résultats dans `bytes_one_token` et
    #       `bytes_full_context`.

    # Assert 1 — le pas de croissance : un token de contexte coûte 12 KiB
    assert bytes_one_token == 12_288
    assert bytes_one_token == 12 * 1024

    # Assert 2 — le contexte complet de 1024 tokens, soit 12 MiB tout rond
    assert bytes_full_context == 12_582_912
    assert bytes_full_context == 12 * 1024**2
    assert bytes_full_context == 1024 * bytes_one_token

    # Assert 3 — le dtype multiplie tout : FP32 coûte exactement le double de FP16
    assert (
        kv_cache_bytes(num_layers, num_kv_heads, head_dim, context_tokens, torch.float32)
        == 25_165_824
    )

    # Assert 4 — `num_heads` n'apparaît pas dans la formule : seules les têtes KV comptent
    assert kv_cache_bytes(24, 2, 64, 1024, torch.float16) == 12_582_912


@pytest.mark.tdd
def test_kv_cache_memory_scales_linearly_with_context_length():
    """Roadmap 5.3 — le cache grandit linéairement avec le contexte, sans plafond.

    Objectif d'apprentissage
    ------------------------
    Dans la formule de 5.2, `num_tokens` est le seul terme qui bouge pendant une génération :
    tous les autres sont figés par l'architecture. La mémoire du cache est donc une droite
    qui passe par l'origine — doubler le contexte double les octets, exactement, sans effet
    de seuil ni d'amortissement.

    C'est la contrainte structurante d'un moteur d'inférence : les poids coûtent un montant
    FIXE (1 Go en FP16, section 5.1), le cache coûte un montant PROPORTIONNEL au contexte.
    Sur les longs contextes le cache finit par dominer, et c'est lui qui fixe le nombre de
    requêtes concurrentes, donc le débit du serveur. C'est aussi ce qui motive les
    optimisations de la suite de la roadmap (paged attention, quantification du cache).

    Schéma mental
    -------------
        Qwen2.5-0.5B, FP16 : 12 288 octets par token (5.2)

              0 token  ->            0 octet
           1024 tokens ->   12 582 912 octets  =  12 MiB
           2048 tokens ->   25 165 824 octets  =  24 MiB   (x2)
           4096 tokens ->   50 331 648 octets  =  48 MiB   (x4)

        octets(2n) = 2 x octets(n)   pour tout n : c'est une droite, pas une courbe

    Ce que ce test vérifie
    ----------------------
    1. le cas de base, un cache vide à 0 token pèse 0 octet (la droite passe par l'origine) ;
    2. les trois valeurs attendues pour 1024, 2048 et 4096 tokens ;
    3. le doublement exact : passer de 1024 à 2048 tokens multiplie la mémoire par 2, et
       passer à 4096 la multiplie par 4 ;
    4. l'additivité qui en découle : le cache de 1024 + 1024 tokens vaut celui de 2048.

    API à faire émerger (cible roadmap : `src/inference_lab/calculators/kv_cache_memory.py`)
    ---------------------------------------------------------------------------------------
        kv_cache_bytes(num_layers, num_kv_heads, head_dim, num_tokens, dtype) -> int

        La roadmap indique seulement « calculator » pour ce test : on réutilise ici le module
        concret de 5.2, aucune API supplémentaire n'est nécessaire.

    Indice : rien à écrire si 5.2 est déjà vert, le test doit passer tel quel — c'est le but
    d'un test de propriété. Si un `num_tokens` de 0 lève une exception ou renvoie autre chose
    que 0, c'est un cas limite mal traité, pas une subtilité de la formule.
    """

    pytest.skip("Roadmap TDD 5.3 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.calculators.kv_cache_memory import kv_cache_bytes

    # Arrange — toujours la configuration Qwen2.5-0.5B (`num_layers` = 24, `num_kv_heads` = 2,
    #           `head_dim` = 64, FP16) et trois longueurs de contexte qui doublent :
    #           `tokens_1k` = 1024, `tokens_2k` = 2048, `tokens_4k` = 4096.

    # Act — demander à l'API les octets du cache pour un contexte vide puis pour les trois
    #       longueurs, et stocker les résultats dans `bytes_empty`, `bytes_1k`, `bytes_2k` et
    #       `bytes_4k`.

    # Assert 1 — la droite passe par l'origine : pas de token, pas d'octet
    assert bytes_empty == 0

    # Assert 2 — les trois valeurs attendues, calculées à la main
    assert bytes_1k == 12_582_912
    assert bytes_2k == 25_165_824
    assert bytes_4k == 50_331_648

    # Assert 3 — doubler les tokens double la mémoire, exactement
    assert bytes_2k == 2 * bytes_1k
    assert bytes_4k == 4 * bytes_1k
    assert bytes_2k / bytes_1k == pytest.approx(2.0, rel=1e-12)

    # Assert 4 — la linéarité rend le calcul additif par tranches de contexte
    assert bytes_1k + bytes_1k == bytes_2k


@pytest.mark.tdd
def test_reducing_kv_heads_reduces_kv_cache_memory_proportionally():
    """Roadmap 5.4 — GQA : diviser les têtes KV divise le cache dans le même rapport.

    Objectif d'apprentissage
    ------------------------
    Grouped-Query Attention garde toutes les têtes de requête mais partage un petit nombre
    de têtes clé/valeur entre elles. Comme `num_kv_heads` est un facteur direct de la
    formule de 5.2, le gain est exactement proportionnel :

        octets_MHA / octets_GQA = num_heads / num_kv_heads

    Qwen2.5-0.5B a 14 têtes de requête pour 2 têtes KV : chaque tête KV est partagée par 7
    têtes de requête, et le cache est 7 fois plus petit qu'en attention multi-têtes
    classique. C'est l'optimisation mémoire la moins chère du modèle — elle ne change ni le
    nombre de paramètres du MLP, ni le nombre de FLOPs de la section 5.6, seulement le
    volume à stocker et à relire. Le test 4.8 vérifie la contrepartie côté shapes : le cache
    ne stocke QUE les têtes KV.

    Schéma mental
    -------------
        num_layers=24, head_dim=64, FP16, 1024 tokens de contexte

            MHA  : num_kv_heads = 14  ->  88 080 384 octets  = 84 MiB
            GQA  : num_kv_heads =  2  ->  12 582 912 octets  = 12 MiB
            MQA  : num_kv_heads =  1  ->   6 291 456 octets  =  6 MiB

        ratio MHA / GQA = 14 / 2 = 7        économie = 75 497 472 octets = 72 MiB

    Ce que ce test vérifie
    ----------------------
    1. les deux valeurs attendues : 88 080 384 octets en MHA (14 têtes KV) contre
       12 582 912 en GQA (2 têtes KV) ;
    2. le ratio est exactement `num_heads / num_kv_heads`, soit 7, et l'économie absolue
       vaut 75 497 472 octets ;
    3. le cas extrême MQA (une seule tête KV) coûte la moitié de GQA : la proportionnalité
       vaut sur toute la plage, pas seulement au point de fonctionnement du modèle ;
    4. GQA ne touche que le cache : à `num_kv_heads` réduit, la mémoire des poids de 5.1
       reste celle du modèle, 1 000 000 000 octets en FP16.

    API à faire émerger (cible roadmap : `src/inference_lab/calculators/kv_cache_memory.py`)
    ---------------------------------------------------------------------------------------
        kv_cache_bytes(num_layers, num_kv_heads, head_dim, num_tokens, dtype) -> int
        def gqa_cache_reduction_ratio(num_heads: int, num_kv_heads: int) -> float: ...

        La roadmap indique « calculator » : on propose de compléter le module concret de 5.2
        avec le helper de ratio, plutôt que de créer un nouveau fichier.

    Indice : `num_heads` doit être un multiple de `num_kv_heads` (sinon le partage des têtes
    est impossible) — c'est la validation à écrire dans le helper. Piège : ne divise pas
    `head_dim` en passant à GQA, il reste à 64 ; c'est le NOMBRE de têtes KV qui change, pas
    leur taille.
    """

    pytest.skip("Roadmap TDD 5.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.calculators.kv_cache_memory import (
        gqa_cache_reduction_ratio,
        kv_cache_bytes,
    )
    from inference_lab.calculators.model_memory import parameter_memory_bytes

    # Arrange — la configuration Qwen2.5-0.5B : `num_layers` = 24, `head_dim` = 64, FP16,
    #           `context_tokens` = 1024, `num_heads` = 14 têtes de requête. Trois variantes
    #           d'attention, décrites par leur seul nombre de têtes KV : `mha_kv_heads` = 14
    #           (une tête KV par tête de requête), `gqa_kv_heads` = 2 (le vrai réglage du
    #           modèle) et `mqa_kv_heads` = 1.

    # Act — demander à l'API les octets du cache pour les trois variantes, à toutes choses
    #       égales par ailleurs, et stocker les résultats dans `mha_bytes`, `gqa_bytes` et
    #       `mqa_bytes`.

    # Assert 1 — les deux points de comparaison, calculés à la main
    assert mha_bytes == 88_080_384
    assert gqa_bytes == 12_582_912

    # Assert 2 — le gain est exactement le rapport des têtes KV
    assert mha_bytes == 7 * gqa_bytes
    assert gqa_cache_reduction_ratio(num_heads, gqa_kv_heads) == pytest.approx(7.0, rel=1e-12)
    assert mha_bytes - gqa_bytes == 75_497_472

    # Assert 3 — la proportionnalité tient jusqu'au cas extrême MQA
    assert mqa_bytes == 6_291_456
    assert gqa_bytes == 2 * mqa_bytes

    # Assert 4 — GQA n'allège que le cache, pas les poids du modèle
    assert parameter_memory_bytes(500_000_000, torch.float16) == 1_000_000_000
