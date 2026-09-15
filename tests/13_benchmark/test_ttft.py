"""Section 13.1 — TTFT : le temps d'attente avant le premier token.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/13_benchmark/test_ttft.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(20 ms de tokenisation, 180 ms de prefill, donc 200 ms de TTFT) : c'est volontaire. Un test
doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

RÈGLE D'OR, reprise de la section 7 : on n'asserte JAMAIS une durée absolue mesurée sur la
machine. Ici la mécanique est rendue déterministe par une HORLOGE INJECTÉE qui rejoue des
instants connus (même convention que `run_benchmark` en 7.1) : le TTFT attendu devient alors
une valeur exacte, écrite en dur dans le test.

Cette section 13 est la couche « métriques » : elle transforme des instants et des compteurs
en indicateurs publiables (TTFT, TPOT, débit, mémoire). Le protocole de mesure lui-même
(warmup, répétitions, médiane) appartient à la section 7 et n'est pas redéfini ici.

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
def test_ttft_metric_measures_time_until_first_token():
    """Roadmap 13.1 — le TTFT est la latence perçue par l'utilisateur, et c'est le prefill.

    Objectif d'apprentissage
    ------------------------
    Le TTFT (*time to first token*) est le temps entre l'arrivée de la requête et l'instant
    où le premier token s'affiche. C'est la seule métrique que l'utilisateur ressent comme
    « le temps de réponse » : tant qu'aucun token n'est sorti, l'écran est vide, et un chatbot
    à 2 s de TTFT est jugé lent même s'il génère ensuite 100 tokens/s. Le reste de la
    génération, lui, est perçu comme une vitesse de défilement, pas comme une attente
    (c'est le TPOT, section 13.2).

    Ce que le TTFT contient :

        TTFT = tokenisation du prompt + prefill (1 forward sur les N tokens du prompt)

    La tokenisation est négligeable (quelques millisecondes de CPU), donc le TTFT est
    DOMINÉ par le prefill. Or le prefill est *compute-bound* (section 5.10) : il traite les N
    positions du prompt d'un coup, les GEMM travaillent sur des matrices (N, hidden) et le
    coût croît avec la longueur du prompt. Conséquences pratiques : pour améliorer le TTFT on
    joue sur le calcul (prompt plus court, prefix caching de 14.7, chunked prefill de 14.8),
    pas sur la bande passante — à l'inverse exact du decode.

    Schéma mental
    -------------
        horloge injectée, instants en secondes :

            t = 0.00   requête reçue          --> lecture 1
            t = 0.02   prompt tokenisé        --> lecture 2   (20 ms, part négligeable)
            t = 0.20   1er token disponible   --> lecture 3   (180 ms de prefill)

            tokenize = 0.02 - 0.00 = 0.02 s
            prefill  = 0.20 - 0.02 = 0.18 s      <- 90 % du TTFT
            TTFT     = 0.20 - 0.00 = 0.20 s      <- ce que l'utilisateur attend

        3 lectures d'horloge pour 2 phases : un seul chronomètre, deux bornes intermédiaires

    Ce que ce test vérifie
    ----------------------
    1. les deux phases sont chronométrées séparément et dans l'ordre : la tokenisation est
       appelée une fois avant le prefill, lui aussi appelé une fois, et l'horloge est lue
       exactement 3 fois ;
    2. le TTFT vaut exactement 0,20 s, la valeur injectée, et il est égal à la somme des deux
       phases : aucune durée n'est oubliée entre les deux ;
    3. le prefill représente 90 % du TTFT et pèse plus de huit fois la tokenisation : c'est
       bien le prefill qu'il faut optimiser pour améliorer la latence perçue ;
    4. le TTFT ne dit rien du reste de la génération : il s'arrête au premier token, donc il
       est strictement plus grand que le prefill seul mais indépendant du nombre de tokens
       générés ensuite.

    API à faire émerger (la roadmap dit seulement « metrics », cible proposée :
    `src/inference_lab/metrics/latency.py`)
    -------------------------------------------------------------------------
        @dataclass(frozen=True)
        class TimeToFirstToken:
            tokenize_seconds: float
            prefill_seconds: float
            time_to_first_token_seconds: float

        def measure_time_to_first_token(
            tokenize: Callable[[], object],
            prefill: Callable[[], object],
            *,
            clock: Callable[[], float] = time.perf_counter,
        ) -> TimeToFirstToken: ...

    Indice : trois lectures d'horloge encadrent les deux appels
    (`start`, `after_tokenize`, `after_prefill`), et le TTFT est `after_prefill - start`, pas
    la somme recalculée des deux phases. Le paramètre `clock` n'existe que pour la testabilité
    (`time.perf_counter` en production, jamais `time.time` qui n'est pas monotone) : c'est la
    convention déjà posée par `run_benchmark` en 7.1. Pièges : ne rappelle pas `clock()` deux
    fois entre les phases (tu créerais un trou non mesuré), et sur GPU il faut
    `torch.cuda.synchronize()` avant chaque lecture, sinon tu mesures la file de lancement des
    kernels et non le calcul (section 6.2).
    """

    pytest.skip("Roadmap TDD 13.1 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.metrics.latency import measure_time_to_first_token

    # Arrange — pas de vrai modèle : deux fonctions jouets instrumentées et une horloge
    #           injectée. Construire `calls`, une liste vide servant de journal d'appels ;
    #           `toy_tokenize` et `toy_prefill`, deux fonctions sans argument qui ajoutent
    #           chacune leur nom (`"tokenize"` puis `"prefill"`) à `calls` ; `clock_reads`, une
    #           liste vide ; et `fake_clock`, une horloge factice sans argument qui rejoue,
    #           lecture après lecture, les instants 0,00 s puis 0,02 s puis 0,20 s en
    #           enregistrant chaque valeur rendue dans `clock_reads`.

    # Act — demander à l'API le TTFT de ce couple tokenisation / prefill en lui passant
    #       l'horloge factice, et garder le résultat dans `ttft`.

    # Assert 1 — les deux phases sont exécutées dans l'ordre, l'horloge lue 3 fois
    assert calls == ["tokenize", "prefill"]
    assert clock_reads == [0.0, 0.02, 0.2]

    # Assert 2 — le TTFT est la valeur injectée, et il couvre les deux phases sans trou
    assert ttft.time_to_first_token_seconds == pytest.approx(0.2, rel=1e-12)
    assert ttft.tokenize_seconds == pytest.approx(0.02, rel=1e-12)
    assert ttft.prefill_seconds == pytest.approx(0.18, rel=1e-9)
    assert ttft.time_to_first_token_seconds == pytest.approx(
        ttft.tokenize_seconds + ttft.prefill_seconds, rel=1e-9
    )

    # Assert 3 — le TTFT est dominé par le prefill : 90 % du total, et non la tokenisation
    assert ttft.prefill_seconds / ttft.time_to_first_token_seconds == pytest.approx(0.9, rel=1e-9)
    assert ttft.prefill_seconds > 8 * ttft.tokenize_seconds

    # Assert 4 — la mesure s'arrête au premier token : rien du decode n'y entre
    assert ttft.time_to_first_token_seconds > ttft.prefill_seconds
    assert isinstance(ttft.time_to_first_token_seconds, float)
