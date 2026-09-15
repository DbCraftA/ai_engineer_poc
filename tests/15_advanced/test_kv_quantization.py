"""Section 15.3 — quantifier le KV cache : la mémoire qui limite la concurrence.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/15_advanced/test_kv_quantization.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(12 582 912 octets en FP16, 6 291 456 en INT8, 1024 requêtes concurrentes contre 2048) :
c'est volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même
formule que le code testé.

Aucune formule nouvelle ici : on réutilise telle quelle celle de la section 5.2, en ne
changeant QUE le nombre d'octets par élément (section 1.8) :

    octets_cache = 2 x num_layers x num_kv_heads x head_dim x num_tokens x octets_par_élément

Le facteur 2 compte K et V. La quantification du cache est la suite logique de 15.1 : même
schéma `valeurs INT8 + échelle`, mais appliqué à des données qui grandissent à chaque token
au lieu d'être figées au chargement du modèle.

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
def test_quantized_kv_cache_reduces_cache_memory():
    """Roadmap 15.3 — un cache INT8 tient deux fois plus de requêtes sur la même carte.

    Objectif d'apprentissage
    ------------------------
    Les poids de 15.1 coûtent un montant FIXE : une fois le modèle chargé, l'affaire est
    close. Le KV cache, lui, coûte un montant PROPORTIONNEL au contexte ET au nombre de
    requêtes servies en parallèle (section 5.3). C'est donc lui qui fixe le nombre de
    séquences qu'un serveur peut tenir en vol, donc son débit :

        requêtes_concurrentes = (mémoire_disponible - poids) / octets_de_cache_par_requête

    Quantifier le cache en INT8 divise le dénominateur par deux et double directement cette
    capacité. C'est le même geste qu'en 15.1 — `octets_par_élément` passe de 2 à 1 dans la
    formule de 5.2 — mais l'enjeu n'est pas la bande passante, c'est la CAPACITÉ : un cache
    deux fois plus petit, c'est deux fois plus de clients servis avant le premier OOM, ou un
    contexte deux fois plus long pour le même client.

    Les métadonnées ne disparaissent pas pour autant. Un cache se quantifie par petits
    blocs — typiquement une échelle par (K ou V, couche, tête KV, token) — sinon un seul
    token aberrant dégraderait tout le contexte. Ces échelles se paient en octets et rognent
    le facteur 2 : c'est le prix de la granularité vu en 15.1.

    Schéma mental
    -------------
        Qwen2.5-0.5B : num_layers=24, num_kv_heads=2, head_dim=64, batch=1, 1024 tokens

            FP32 : 2 x 24 x 2 x 64 x 1024 x 4  =  25 165 824 octets  = 24 MiB
            FP16 : 2 x 24 x 2 x 64 x 1024 x 2  =  12 582 912 octets  = 12 MiB
            INT8 : 2 x 24 x 2 x 64 x 1024 x 1  =   6 291 456 octets  =  6 MiB

            échelles FP16, une par (K/V, couche, tête, token) :
                2 x 24 x 2 x 1024 = 98 304 échelles x 2 =    196 608 octets
                total INT8 réel                         =  6 488 064 octets

        budget de cache de 12 GiB :
            FP16 :  12 GiB / 12 MiB = 1024 requêtes
            INT8 :  12 GiB /  6 MiB = 2048 requêtes      <- capacité doublée
            INT8 + échelles         = 1985 requêtes      <- le prix des métadonnées

    Ce que ce test vérifie
    ----------------------
    1. les références de la section 5.2, inchangées : 12 582 912 octets en FP16 et
       25 165 824 en FP32 pour 1024 tokens ;
    2. le même cache en INT8 pèse 6 291 456 octets, soit exactement la moitié du FP16 et le
       quart du FP32 — le facteur ne dépend que du nombre d'octets par élément ;
    3. les échelles par bloc ajoutent 196 608 octets (98 304 échelles FP16), ce qui ramène le
       gain de 2,000 à 1,939 ;
    4. la conséquence qui compte : à budget de cache constant de 12 GiB, on passe de 1024 à
       2048 requêtes concurrentes (1985 en comptant les échelles).

    API à faire émerger (cible roadmap « future KV quantization », cible proposée :
    `src/inference_lab/quantization/kv_cache.py`, nouveau package `quantization`)
    -----------------------------------------------------------------------------
        def quantized_kv_cache_bytes(
            num_layers: int,
            num_kv_heads: int,
            head_dim: int,
            num_tokens: int,
            dtype: torch.dtype = torch.int8,
            scale_dtype: torch.dtype | None = None,
            batch_size: int = 1,
        ) -> int: ...

        def max_concurrent_requests(budget_bytes: int, bytes_per_request: int) -> int: ...

        `scale_dtype=None` ne compte que le corps du cache ; `scale_dtype=torch.float16`
        ajoute une échelle par (K ou V, couche, tête KV, token). La référence FP16 reste
        `kv_cache_bytes` de `src/inference_lab/calculators/kv_cache_memory.py` (section 5.2) :
        le nouveau module ne doit pas dupliquer la formule, seulement la paramétrer.

    Indice : `max_concurrent_requests` renvoie un `int` — c'est une division ENTIÈRE, une
    requête à moitié logée n'existe pas. Piège : `head_dim` n'intervient pas dans le nombre
    d'échelles (une échelle couvre les 64 valeurs d'une tête pour un token), donc les
    métadonnées ne se déduisent pas d'une règle de trois sur les octets du corps.
    """

    pytest.skip("Roadmap TDD 15.3 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.calculators.kv_cache_memory import kv_cache_bytes
    from inference_lab.quantization.kv_cache import (
        max_concurrent_requests,
        quantized_kv_cache_bytes,
    )

    # Arrange — la configuration de référence de Qwen2.5-0.5B, en entiers : `num_layers` = 24,
    #           `num_kv_heads` = 2, `head_dim` = 64, `batch_size` = 1, `context_tokens` = 1024,
    #           et un budget mémoire réservé au cache `budget_bytes` de 12 GiB. Aucun tenseur
    #           à allouer : on ne calcule que des formules fermées.

    # Act — relever les octets du cache pour ce contexte en FP32 (`fp32_bytes`), en FP16
    #       (`fp16_bytes`), en INT8 sans métadonnées (`int8_bytes`) et en INT8 avec une échelle
    #       FP16 par bloc (`int8_with_scales_bytes`), puis le nombre de requêtes concurrentes
    #       tenant dans `budget_bytes` pour ces trois derniers cas : `concurrent_fp16`,
    #       `concurrent_int8` et `concurrent_int8_with_scales`.

    # Assert 1 — les références de la section 5.2 sont inchangées
    assert fp16_bytes == 12_582_912
    assert fp16_bytes == 12 * 1024**2
    assert fp32_bytes == 25_165_824
    assert fp16_bytes == kv_cache_bytes(
        num_layers, num_kv_heads, head_dim, context_tokens, torch.float16
    )

    # Assert 2 — INT8 : moitié du FP16, quart du FP32, exactement
    assert int8_bytes == 6_291_456
    assert int8_bytes == 6 * 1024**2
    assert 2 * int8_bytes == fp16_bytes
    assert 4 * int8_bytes == fp32_bytes

    # Assert 3 — les échelles par bloc rognent le gain : 1,939 au lieu de 2,000
    assert int8_with_scales_bytes - int8_bytes == 196_608
    assert int8_with_scales_bytes == 6_488_064
    assert fp16_bytes / int8_with_scales_bytes == pytest.approx(64 / 33, rel=1e-9)
    assert fp16_bytes / int8_with_scales_bytes < 2.0

    # Assert 4 — la capacité doublée, qui est le vrai objectif
    assert budget_bytes == 12_884_901_888
    assert concurrent_fp16 == 1024
    assert concurrent_int8 == 2048
    assert concurrent_int8 == 2 * concurrent_fp16
    assert concurrent_int8_with_scales == 1985
    assert max_concurrent_requests(budget_bytes, fp16_bytes) == 1024
