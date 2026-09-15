"""Section 13.2 — TPOT : le temps par token de sortie, hors premier token.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/13_benchmark/test_tpot.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(160 ms de decode pour 4 tokens, donc 40 ms par token et 25 tokens/s) : c'est volontaire. Un
test doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code
testé.

RÈGLE D'OR, reprise de la section 7 : aucune durée absolue mesurée n'est assertée. Les
instants d'arrivée des tokens sont INJECTÉS (ce sont les lectures d'une horloge déterministe,
même convention qu'en 7.1), ce qui rend le TPOT attendu exactement calculable et permet de
l'écrire en dur.

Cette métrique est une FORMULE pure sur des instants déjà relevés : elle ne chronomètre rien
elle-même, contrairement à `benchmark_decode` de la section 7.5 qui, lui, exécute la boucle
de decode. Section 13 = calcul de l'indicateur, section 7 = protocole de mesure.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices. Même raison pour l'import
# du module cible : c'est ton code d'`Act` qui l'appellera, le linter le voit donc inutilisé.
# ruff: noqa: F401, F821

import pytest


@pytest.mark.tdd
@pytest.mark.perf
def test_tpot_metric_measures_decode_time_per_output_token():
    """Roadmap 13.2 — le premier token appartient au prefill, pas au decode.

    Objectif d'apprentissage
    ------------------------
    Le TPOT (*time per output token*, aussi appelé ITL, *inter-token latency*) est la vitesse
    de défilement du texte pendant le streaming :

        TPOT = temps total de decode / nombre de tokens générés APRÈS le premier

    Le piège est dans le dénominateur. Le premier token est produit par le prefill : il est
    déjà comptabilisé dans le TTFT (13.1). Si on le recompte dans le decode, on divise par
    N au lieu de N-1 et on annonce un TPOT trop optimiste — d'autant plus faux que la
    génération est courte. Formulé autrement : entre 5 tokens affichés il n'y a que 4
    intervalles.

    Ce chiffre est celui qui bouge quand on optimise la mémoire, parce que le decode est
    *memory-bound* (section 5.10) : chaque étape ne traite qu'UNE position mais relit tous les
    poids et tout le KV cache. C'est donc lui qui réagit à la quantification (15.1), à GQA, au
    KV cache (section 4) ; le TTFT, compute-bound, réagit à d'autres leviers. Publier les deux
    séparément est indispensable : un moteur peut améliorer l'un en dégradant l'autre.

    Schéma mental
    -------------
        instants d'arrivée des tokens (secondes), injectés :

            token 1 : 0.20   <- fin du prefill, c'est le TTFT (13.1), PAS du decode
            token 2 : 0.24   ) 4 intervalles de 40 ms
            token 3 : 0.28   )
            token 4 : 0.32   )
            token 5 : 0.36   )

            temps de decode = 0.36 - 0.20 = 0.16 s
            tokens comptés  = 5 - 1 = 4
            TPOT            = 0.16 / 4 = 0.04 s = 40 ms
            tokens/s        = 1 / 0.04 = 25 tokens/s

        erreur classique : 0.16 / 5 = 0.032 s, soit un TPOT sous-estimé de 20 %

    Ce que ce test vérifie
    ----------------------
    1. le décompte : 5 instants d'arrivée valent 4 tokens de decode et un temps de decode de
       0,16 s, le premier token restant hors du chronomètre du decode ;
    2. le TPOT vaut exactement 0,04 s, et il est strictement supérieur à la valeur erronée
       0,032 s qu'on obtiendrait en comptant le premier token ;
    3. la relation exacte `tokens_per_second == 1 / tpot`, soit 25 tokens/s ;
    4. la formule marche aussi directement sur un total déjà agrégé (0,16 s pour 4 tokens) et
       reste invariante si l'on double simultanément durée et nombre de tokens : c'est un
       temps PAR token, pas un temps total.

    API à faire émerger (la roadmap dit seulement « metrics », cible proposée :
    `src/inference_lab/metrics/latency.py`, module déjà visé par 13.1)
    -------------------------------------------------------------------------
        def decode_seconds_from_timestamps(token_timestamps: Sequence[float]) -> float: ...
        def decoded_token_count(token_timestamps: Sequence[float]) -> int: ...
        def time_per_output_token(decode_seconds: float, generated_tokens: int) -> float: ...
        def output_tokens_per_second(time_per_output_token_seconds: float) -> float: ...

    Indice : `timestamps[-1] - timestamps[0]` pour la durée, `len(timestamps) - 1` pour le
    nombre d'intervalles, puis une simple division qui doit refuser `generated_tokens == 0`
    (une requête d'un seul token n'a pas de TPOT : il n'y a aucun intervalle à mesurer).
    Pièges : ne réimplémente pas ici la boucle de decode chronométrée, elle existe déjà en
    7.5 (`benchmark_decode`) ; et n'agrège pas les intervalles avec une moyenne des écarts
    successifs sans raison — la définition publiée est bien « durée totale / tokens », donc
    robuste à un intervalle isolé anormalement lent.
    """

    pytest.skip("Roadmap TDD 13.2 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.metrics.latency import (
        decode_seconds_from_timestamps,
        decoded_token_count,
        output_tokens_per_second,
        time_per_output_token,
    )

    # Arrange — que des nombres, aucun modèle ni horloge réelle. Construire
    #           `token_timestamps`, la liste des instants d'arrivée des 5 tokens d'une réponse,
    #           en secondes : le premier à la fin du prefill (0,20 s), puis un token toutes les
    #           40 ms jusqu'à 0,36 s. Ces instants sont ceux qu'aurait rendus une horloge
    #           monotone injectée, donc le résultat attendu est connu à l'avance.

    # Act — demander à l'API la durée de decode et le nombre de tokens décodés déduits de
    #       `token_timestamps`, puis en tirer le TPOT et le débit de sortie. Stocker les
    #       résultats dans `decode_seconds`, `generated_tokens`, `tpot` et `tokens_per_second`.

    # Assert 1 — 5 instants = 4 tokens de decode, et le 1er token reste hors du chronomètre
    assert generated_tokens == 4
    assert decode_seconds == pytest.approx(0.16, rel=1e-9)

    # Assert 2 — le TPOT chiffré, et le piège du premier token compté en trop
    assert tpot == pytest.approx(0.04, rel=1e-9)
    assert tpot > 0.032

    # Assert 3 — relation exacte entre temps par token et tokens par seconde
    assert tokens_per_second == pytest.approx(25.0, rel=1e-9)
    assert tokens_per_second == pytest.approx(1.0 / tpot, rel=1e-12)

    # Assert 4 — c'est un temps PAR token : mêmes valeurs sur un total agrégé, invariant
    #            si l'on double durée et tokens ensemble
    assert time_per_output_token(0.16, 4) == pytest.approx(0.04, rel=1e-12)
    assert time_per_output_token(0.32, 8) == pytest.approx(0.04, rel=1e-12)
    assert output_tokens_per_second(0.04) == pytest.approx(25.0, rel=1e-12)
