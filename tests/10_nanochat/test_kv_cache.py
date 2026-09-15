"""Section 10.5 — validation de bout en bout : la génération avec cache est-elle identique ?

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/10_nanochat/test_kv_cache.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(ids (1, 10), logits (1, 4, 32), 9 positions traitées contre 30) : c'est volontaire. Un test
doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Modèle jouet de la section 10, constant dans tout le chapitre : `vocab_size=32`,
`hidden_size=16`, `num_layers=2`, `num_heads=4`, `num_kv_heads=2`, `head_dim=4`,
`intermediate_size=32`, `seq=6` (longueur du prompt), `batch=1`. C'est le test final du
chapitre : il branche ensemble tout ce que la roadmap a construit — l'architecture moderne de
10.2, la boucle de génération de 3.10/10.4 et le KV cache de la section 4 — sur un modèle qui
ressemble enfin à un vrai LLM. Le KV cache y est bien plus intéressant qu'en section 4 : avec
GQA (`num_kv_heads=2` contre `num_heads=4`) le cache ne stocke que la moitié des têtes, et
avec RoPE la position du token décodé doit être calculée à la main puisque le modèle ne reçoit
qu'un seul id à la fois.

Une optimisation qui change le résultat n'est pas une optimisation, c'est un bug. L'égalité
attendue est donc double : EXACTE sur les ids greedy (`torch.equal`), et numérique sur les
logits (`rtol=1e-5, atol=1e-6`) puisque les sommes flottantes ne se font pas dans le même
ordre selon qu'on recalcule ou qu'on concatène les K/V.

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
@pytest.mark.model
def test_nanochat_like_cached_generation_matches_uncached_generation():
    """Roadmap 10.5 — sur une architecture complète aussi, le cache change le coût, pas le résultat.

    Objectif d'apprentissage
    ------------------------
    Les K/V d'un token ne dépendent pas des tokens suivants : les mémoriser ou les recalculer
    donne le même vecteur. Cette identité est ce qui autorise le KV cache, et elle est
    vérifiable — d'où ce test, qui est la validation de bout en bout du chapitre.

    Sur cette architecture, deux détails rendent l'égalité fragile, et c'est précisément
    pourquoi il faut la tester ici plutôt que sur le modèle jouet de la section 3. RoPE
    d'abord : au decode le modèle ne reçoit qu'un id, donc plus rien dans son entrée ne dit OÙ
    ce token se trouve ; la position doit venir de la longueur du cache (section 4.7). Si elle
    est laissée à zéro, les logits divergent, le texte reste plausible quelques tokens puis
    part en boucle. GQA ensuite : le cache stocke `num_kv_heads = 2` têtes, mais l'attention
    en consomme `num_heads = 4` ; la duplication des têtes KV doit se faire dans le même ordre
    que dans le chemin sans cache, sinon les têtes de requêtes sont appariées aux mauvais
    groupes.

    Le gain, lui, est structurel : sans cache chaque itération repaie tout le préfixe, avec
    cache elle ne paie qu'un token. C'est la différence entre un coût quadratique et un coût
    linéaire en nombre de tokens générés.

    Schéma mental
    -------------
        batch=1, num_layers=2, num_heads=4, num_kv_heads=2, head_dim=4, vocab=32
        prompt de 6 tokens, 4 tokens générés en greedy

        sans cache : [t0..t5] -> [t0..t6] -> [t0..t7] -> [t0..t8]
                     6 + 7 + 8 + 9 = 30 positions traitées
        avec cache : [t0..t5] -> [t6]     -> [t7]     -> [t8]
                     6 + 1 + 1 + 1 =  9 positions traitées

        ids (1, 10) IDENTIQUES         step_logits (1, 4, 32) égaux à 1e-5 près
        cache par couche : (1, num_kv_heads=2, seq, head_dim=4), pas 4 têtes

    Ce que ce test vérifie
    ----------------------
    1. les deux chemins rendent des sorties de mêmes shapes : ids `(1, 10)` en `torch.long`
       et logits par étape `(1, 4, 32)` ;
    2. les ids greedy sont EXACTEMENT égaux (`torch.equal`, sans tolérance) et le prompt reste
       intact en préfixe des 4 tokens générés ;
    3. les logits de chaque étape coïncident à la tolérance float32 (`rtol=1e-5, atol=1e-6`),
       sur des valeurs réellement finies et en float32 — sinon la comparaison ne démontrerait
       rien ;
    4. le coût, lui, diffère franchement : 9 positions traitées avec cache contre 30 sans.

    API à faire émerger (cible roadmap « cache/inference », cibles proposées :
    `src/inference_lab/inference/generation.py` et `src/inference_lab/cache/kv_cache.py`,
    modules déjà visés par 3.10/4.6 et 4.2)
    ------------------------------------------------------------------------------------
        output_ids, step_logits = generate(
            model, prompt_ids, max_new_tokens, use_cache=True, return_step_logits=True
        )

    Aucune API nouvelle : `generate` (10.4) et `KVCache` (4.2) suffisent. Le travail de cette
    section est de faire cohabiter les deux sur le modèle de 10.2, en passant le cache et la
    position du token courant à chaque bloc.

    Indice : une seule boucle avec deux branches internes — `use_cache=False` repasse tous les
    ids, `use_cache=True` ne passe que le dernier id avec le cache. Pièges : la position RoPE
    du token décodé vaut `cache.seq_len` AVANT l'ajout ; le masque causal du decode doit
    autoriser tout le préfixe mémorisé (une seule ligne, `seq_len + 1` colonnes) ; le modèle
    doit être en `eval()`, sous `torch.no_grad()` et SANS dropout, sinon les deux chemins
    tirent des masques aléatoires différents et l'égalité est perdue pour une raison qui n'a
    rien à voir avec le cache.
    """

    pytest.skip("Roadmap TDD 10.5 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.inference.generation import generate

    # Arrange — fixer la graine (`torch.manual_seed(0)`) puis construire :
    #           - `model`, le `NanochatLikeModel` de la section 10.2 (vocab_size=32,
    #             hidden_size=16, num_layers=2, num_heads=4, num_kv_heads=2, head_dim=4,
    #             intermediate_size=32) : le MÊME objet pour les deux chemins, donc des poids
    #             identiques, en `eval()`, sans dropout, en float32. Le modèle compte dans
    #             `positions_processed` le nombre total de positions reçues depuis la dernière
    #             remise à zéro ;
    #           - `prompt_ids`, un tenseur `torch.long` de shape (batch=1, seq=6) d'ids
    #             distincts dans [0, 32) ;
    #           - `max_new_tokens = 4`.

    # Act — générer deux fois en greedy avec `generate`, en remettant le compteur du modèle à
    #       zéro avant chaque génération : une fois `use_cache=False` (`uncached_ids`,
    #       `uncached_logits`, compteur relevé dans `uncached_positions`), une fois
    #       `use_cache=True` (`cached_ids`, `cached_logits`, `cached_positions`).

    # Assert 1 — mêmes shapes de part et d'autre
    assert cached_ids.shape == (1, 10)
    assert cached_ids.dtype is torch.long
    assert uncached_ids.shape == cached_ids.shape
    assert cached_logits.shape == (1, 4, 32)
    assert uncached_logits.shape == (1, 4, 32)

    # Assert 2 — égalité EXACTE des ids greedy, prompt intact en préfixe
    assert torch.equal(cached_ids, uncached_ids)
    assert torch.equal(cached_ids[:, :6], prompt_ids)

    # Assert 3 — mêmes logits à la tolérance float32 près, sur de vraies valeurs finies
    assert cached_logits.dtype is torch.float32
    assert bool(torch.isfinite(cached_logits).all())
    assert bool(torch.isfinite(uncached_logits).all())
    torch.testing.assert_close(cached_logits, uncached_logits, rtol=1e-5, atol=1e-6)

    # Assert 4 — même résultat, coût très différent
    assert cached_positions == 9
    assert uncached_positions == 30
