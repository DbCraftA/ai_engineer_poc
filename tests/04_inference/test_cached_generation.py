"""Section 4.6 / 4.7 — le cache doit être transparent : mêmes logits, mêmes tokens.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/04_inference/test_cached_generation.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(logits (1, 3, 16), 6 ids, positions 3 puis 4 puis 5, 5 positions traitées contre 12) :
c'est volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même
formule que le code testé.

C'est la section la plus importante du chapitre 4 : une optimisation qui change le résultat
n'est pas une optimisation, c'est un bug. Le cache est une réécriture du calcul à résultat
constant, donc le test de référence est une comparaison au chemin naïf de 4.1. Les logits se
comparent avec une tolérance float32 (l'ordre des opérations diffère), mais les tokens greedy
doivent être identiques à l'id près. Le test 4.7 isole la cause d'erreur numéro un lorsque
cette égalité échoue : une position RoPE mal calculée au moment du decode.

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
def test_cached_and_uncached_generation_produce_same_logits():
    """Roadmap 4.6 — le KV cache change le coût du calcul, jamais son résultat.

    Objectif d'apprentissage
    ------------------------
    Le cache repose sur une identité mathématique : les K/V d'un token ne dépendent pas des
    tokens suivants, donc les réutiliser ou les recalculer donne le même vecteur. Toute
    différence observée vient d'un bug — masque causal oublié, position RoPE erronée,
    concaténation dans le mauvais ordre — et non du cache lui-même.

    C'est le test de non-régression que l'on rejouera après chaque optimisation de la suite
    de la roadmap (SDPA en section 8, batching en 9, quantification en 10) : une
    implémentation rapide n'a de valeur que si elle reste d'accord avec l'implémentation
    lente de référence.

    L'égalité est numérique, pas binaire : avec cache on calcule `Q @ K_cache.T` sur un
    tenseur concaténé, sans cache sur un tenseur recalculé en une passe. Les sommes flottantes
    ne se font pas dans le même ordre, d'où une tolérance float32 (rtol=1e-5, atol=1e-6) et
    non un `torch.equal`.

    Schéma mental
    -------------
        batch=1, num_layers=2, num_heads=2, num_kv_heads=1, head_dim=4, vocab=16
        prompt de 3 tokens, 3 tokens générés

        sans cache : [t0 t1 t2] -> [t0..t3] -> [t0..t4]     12 positions traitées
        avec cache : [t0 t1 t2] -> [t3]     -> [t4]          5 positions traitées

        step_logits (1, 3, 16) dans les deux cas, égaux à 1e-5 près

    Ce que ce test vérifie
    ----------------------
    1. les deux chemins rendent des logits de même shape (1, 3, 16) et 6 ids en sortie ;
    2. les logits de chaque étape coïncident à la tolérance float32 (rtol=1e-5, atol=1e-6) ;
    3. les valeurs comparées sont finies et en float32 : la comparaison porte sur de vrais
       logits, pas sur des NaN qui rendraient l'assert vide de sens ;
    4. le coût, lui, diffère : 5 positions traitées avec cache contre 12 sans.

    API à faire émerger (cible roadmap « cache + inference », cible proposée :
    `src/inference_lab/inference/generation.py`, module déjà visé par 3.10)
    ------------------------------------------------------------------------
        def generate(
            model: torch.nn.Module,
            prompt_ids: torch.Tensor,
            max_new_tokens: int,
            use_cache: bool = True,
        ) -> tuple[torch.Tensor, torch.Tensor]: ...

    La fonction rend `(output_ids, step_logits)` : les ids complets, et les logits utilisés à
    chaque étape empilés en (batch, max_new_tokens, vocab).

    Indice : une seule boucle avec deux branches internes — `use_cache=False` repasse tous
    les ids, `use_cache=True` ne passe que le dernier id avec le cache. Piège : le modèle doit
    être en `eval()` et sous `torch.no_grad()`, et surtout SANS dropout, sinon les deux
    chemins tirent des masques aléatoires différents et l'égalité est perdue pour une raison
    qui n'a rien à voir avec le cache.
    """

    pytest.skip("Roadmap TDD 4.6 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.inference.generation import generate

    # Arrange — dimensions jouets constantes dans tout le fichier : batch=1, num_layers=2,
    #           num_heads=2, num_kv_heads=1, head_dim=4 (hidden=8), vocab=16, prompt de
    #           3 tokens. Construire `prompt_ids` (shape (1, 3), `torch.long`, ids distincts)
    #           et `model`, le MÊME modèle causal jouet pour les deux chemins : poids
    #           identiques (seed fixée), `eval()`, aucun dropout, float32. Le modèle compte
    #           dans `positions_processed` le nombre total de positions reçues.

    # Act — générer 3 tokens deux fois avec `generate` : une fois `use_cache=False`
    #       (`uncached_ids`, `uncached_logits`, compteur relevé dans `uncached_positions`),
    #       une fois `use_cache=True` (`cached_ids`, `cached_logits`, `cached_positions`), en
    #       remettant le compteur du modèle à zéro avant chaque génération.

    # Assert 1 — mêmes shapes de part et d'autre
    assert cached_logits.shape == (1, 3, 16)
    assert uncached_logits.shape == (1, 3, 16)
    assert cached_ids.shape == (1, 6)

    # Assert 2 — MÊMES logits, à la tolérance float32 près
    torch.testing.assert_close(cached_logits, uncached_logits, rtol=1e-5, atol=1e-6)

    # Assert 3 — la comparaison porte sur de vrais logits finis en float32
    assert cached_logits.dtype is torch.float32
    assert bool(torch.isfinite(cached_logits).all())
    assert bool(torch.isfinite(uncached_logits).all())

    # Assert 4 — même résultat, coût très différent
    assert cached_positions == 5
    assert uncached_positions == 12


@pytest.mark.tdd
def test_cached_and_uncached_greedy_generation_produce_same_tokens():
    """Roadmap 4.6 — en greedy, l'égalité attendue est exacte : mêmes ids de tokens.

    Objectif d'apprentissage
    ------------------------
    L'argmax est une fonction en escalier : tant que le logit gagnant garde une avance
    supérieure au bruit numérique, une différence de 1e-6 sur les logits ne change aucun id.
    C'est pourquoi une génération greedy avec et sans cache doit rendre EXACTEMENT la même
    suite de tokens, comparable avec `torch.equal` sans tolérance.

    C'est aussi l'assertion la plus utile en pratique : elle attrape les bugs de cache qui
    passeraient sous une tolérance trop généreuse sur les logits. La limite à connaître :
    en cas d'égalité parfaite entre deux logits, l'argmax devient arbitraire ; le test
    exige donc explicitement une marge non nulle entre le meilleur et le deuxième logit.

    Schéma mental
    -------------
        prompt (1, 3) + 3 tokens générés -> ids (1, 6) dans les deux cas

        étape 1 : argmax(logits) -> t3       cache : t3       identiques
        étape 2 : argmax(logits) -> t4       cache : t4       identiques
        étape 3 : argmax(logits) -> t5       cache : t5       identiques

        ids[:, :3] == prompt_ids  (le prompt n'est jamais réécrit)

    Ce que ce test vérifie
    ----------------------
    1. les ids générés avec et sans cache sont EXACTEMENT égaux, en shape (1, 6) et en
       dtype entier ;
    2. le prompt est conservé tel quel en préfixe des 3 tokens générés ;
    3. l'argmax est décidé sans ex aequo : à chaque étape, l'écart entre le premier et le
       deuxième logit est strictement positif, ce qui rend l'égalité exacte légitime ;
    4. la génération est reproductible : deux appels identiques rendent les mêmes ids.

    API à faire émerger (cible roadmap « cache + inference », cible proposée :
    `src/inference_lab/inference/generation.py`)
    ------------------------------------------------------------------------
        output_ids, step_logits = generate(model, prompt_ids, max_new_tokens, use_cache)

    Indice : `logits.topk(2, dim=-1).values` donne le meilleur et le deuxième logit d'une
    étape ; leur différence est la marge de l'assert 3. Piège : `torch.testing.assert_close`
    sur des ids entiers masquerait un décalage d'un token, alors que `torch.equal` échoue
    franchement — pour des ids, on veut l'égalité stricte.
    """

    pytest.skip("Roadmap TDD 4.6 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.inference.generation import generate

    # Arrange — mêmes dimensions jouets et même modèle jouet déterministe qu'au test
    #           précédent (batch=1, num_layers=2, num_heads=2, num_kv_heads=1, head_dim=4,
    #           vocab=16, prompt de 3 tokens dans `prompt_ids`), en `eval()` et sans dropout.
    #           Les poids doivent produire des logits DISTINCTS, sans ex aequo en tête.

    # Act — générer 3 tokens en greedy sans cache (`uncached_ids`), avec cache (`cached_ids`),
    #       puis une seconde fois avec cache (`cached_ids_again`) ; conserver les logits par
    #       étape du chemin caché et en déduire `top_margins`, l'écart entre le premier et le
    #       deuxième logit de chaque étape, de shape (1, 3).

    # Assert 1 — égalité EXACTE des ids générés
    assert torch.equal(cached_ids, uncached_ids)
    assert cached_ids.shape == (1, 6)
    assert cached_ids.dtype is torch.long

    # Assert 2 — le prompt est intact en préfixe
    assert torch.equal(cached_ids[:, :3], prompt_ids)

    # Assert 3 — aucun ex aequo : l'argmax est déterminé sans ambiguïté
    assert top_margins.shape == (1, 3)
    assert bool((top_margins > 0.0).all())

    # Assert 4 — génération reproductible
    assert torch.equal(cached_ids, cached_ids_again)


@pytest.mark.tdd
def test_decode_position_advances_with_cache_length():
    """Roadmap 4.7 — au decode, la position RoPE du nouveau token est la longueur du cache.

    Objectif d'apprentissage
    ------------------------
    Au decode, le modèle ne reçoit qu'un token : plus rien dans son entrée ne dit OÙ ce token
    se trouve dans la séquence. L'information de position vient donc du cache, dont la
    longueur courante est exactement l'index du prochain token (indices 0-based : 3 positions
    déjà mémorisées, le nouveau token est en position 3).

    C'est la source du bug le plus fréquent d'un moteur d'inférence maison : oublier
    d'avancer la position, et faire tourner RoPE comme si chaque token généré était le
    premier. Le modèle continue de produire du texte plausible sur quelques tokens, puis
    part en boucle — et les logits divergent du chemin sans cache, ce qui casse 4.6.

    Schéma mental
    -------------
        cache de 3 positions (prompt) -> nouveau token en position 3
        cache de 4 positions          -> nouveau token en position 4
        cache de 5 positions          -> nouveau token en position 5

        position = cache.seq_len AVANT l'ajout, puis cache.seq_len == position + 1 après

        oracle : dans une passe complète sur 4 tokens, RoPE applique aussi la position 3 au
                 dernier token -> mêmes valeurs à 1e-5 près

    Ce que ce test vérifie
    ----------------------
    1. les positions utilisées aux trois étapes de decode sont 3, puis 4, puis 5 ;
    2. la position vaut la longueur du cache AVANT l'ajout, et le cache mesure position + 1
       après ;
    3. le vecteur roté au decode est identique à la dernière position rotée par une passe
       complète (rtol=1e-5, atol=1e-6) ;
    4. la position n'est pas neutre : la même rotation appliquée en position 0 donne un
       résultat différent, donc l'assert 3 démontre bien quelque chose.

    API à faire émerger (cible roadmap « inference », cible proposée :
    `src/inference_lab/inference/decode.py`, module déjà visé par 4.5)
    -----------------------------------------------------------------
        def decode_position(cache: KVCache) -> int: ...

    et la réutilisation de `apply_rope(x, position_ids)` de
    `src/inference_lab/nn/positional/rope.py` (section 2.10).

    Indice : `decode_position` est un one-liner (`return cache.seq_len`) — l'intérêt est de
    nommer l'invariant et de le tester. Piège : ne compte pas les positions dans une variable
    globale ou un compteur d'étapes séparé du cache ; deux sources de vérité finiraient par
    se désynchroniser (reprise de session, troncature de contexte, éviction).
    """

    pytest.skip("Roadmap TDD 4.7 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.inference.decode import decode_position
    from inference_lab.nn.positional.rope import apply_rope

    # Arrange — mêmes dimensions jouets : batch=1, num_layers=2, num_heads=2, num_kv_heads=1,
    #           head_dim=4, prompt de 3 tokens. Construire `cache`, un KVCache déjà rempli
    #           par le prompt (3 positions), et `query`, la requête du token en cours de
    #           decode, shape (1, 2, 1, 4) en float32 (seed fixée). Prévoir aussi
    #           `full_query`, les requêtes des 4 tokens d'une passe complète, shape
    #           (1, 2, 4, 4), dont la dernière ligne est `query`.

    # Act — relever `positions_used`, la liste des positions rendues par `decode_position` aux
    #       trois étapes de decode (le cache grandissant d'une position à chaque étape), puis
    #       appliquer RoPE à `query` en position 3 (`rotated_decode`), à `full_query` sur les
    #       positions 0 à 3 (`rotated_full`), et à `query` en position 0
    #       (`rotated_at_position_zero`).

    # Assert 1 — la position avance d'une unité par étape de decode
    assert positions_used == [3, 4, 5]

    # Assert 2 — position = longueur du cache avant l'ajout ; cache = position + 1 après
    assert positions_used[0] == 3
    assert cache.seq_len == positions_used[-1] + 1
    assert decode_position(cache) == 6

    # Assert 3 — mêmes valeurs qu'une passe complète, où le dernier token est en position 3
    torch.testing.assert_close(rotated_decode, rotated_full[:, :, -1:, :], rtol=1e-5, atol=1e-6)

    # Assert 4 — la position compte vraiment : en position 0, le résultat diffère
    assert not torch.allclose(rotated_decode, rotated_at_position_zero)
