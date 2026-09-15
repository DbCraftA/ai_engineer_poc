"""Section 13.3 — débit : tokens/s d'une requête et tokens/s du système.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/13_benchmark/test_throughput.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(20 tokens/s seul, 50 tokens/s pour un batch de 4, 12,5 tokens/s par requête dans ce batch) :
c'est volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même
formule que le code testé.

RÈGLE D'OR, reprise de la section 7 : aucune durée absolue mesurée n'est assertée. Les temps
écoulés sont INJECTÉS (5 s pour une requête seule, 8 s pour un batch de 4), ce qui rend les
débits attendus exactement calculables et permet de les écrire en dur. Le protocole de mesure
(warmup, répétitions, médiane) reste celui de la section 7.

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
def test_throughput_reports_tokens_per_second():
    """Roadmap 13.3 — batcher dégrade la latence par requête et augmente le débit système.

    Objectif d'apprentissage
    ------------------------
    « tokens/s » désigne deux grandeurs différentes qu'il ne faut jamais confondre :

        débit d'une requête = tokens générés par CETTE requête / son temps écoulé ;
        débit du système    = somme des tokens de TOUTES les requêtes / temps mur total.

    À batch 1 les deux coïncident. Dès qu'on batche, ils divergent en sens opposés, et c'est
    exactement l'intérêt du batching : le decode est *memory-bound* (5.10), il passe son temps
    à relire les poids du modèle depuis la HBM. Ces poids sont lus UNE fois par étape, que le
    batch contienne 1 ou 4 séquences : le travail utile est multiplié par 4 pour presque le
    même trafic mémoire. L'intensité arithmétique monte (5.8), le GPU est mieux occupé, le
    débit système monte fortement — mais chaque requête individuelle attend un peu plus
    longtemps, car son étape de decode est désormais partagée.

    C'est l'arbitrage central des moteurs d'inférence : un fournisseur d'API optimise le débit
    système (coût par million de tokens), un utilisateur interactif optimise la latence de sa
    requête (TTFT en 13.1, TPOT en 13.2). Le *continuous batching* (14.2) existe pour prendre
    le débit du batching sans en payer toute la latence.

    Schéma mental
    -------------
        requête seule (batch 1) : 100 tokens en 5 s
            débit requête = 100 / 5 = 20 tokens/s      = débit système (une seule requête)

        batch de 4 requêtes     : 4 x 100 tokens en 8 s de temps mur
            débit d'une requête = 100 / 8 = 12,5 tokens/s   <- latence dégradée (x0,625)
            débit système       = 400 / 8 = 50 tokens/s     <- x2,5 par rapport à 20

        le temps mur ne quadruple PAS (5 s -> 8 s) : les poids ne sont lus qu'une fois

    Ce que ce test vérifie
    ----------------------
    1. le cas de référence à batch 1 : 100 tokens en 5 s font 20 tokens/s, et le débit système
       d'une requête unique est identique à son débit propre ;
    2. dans un batch de 4 requêtes de 100 tokens terminé en 8 s, chaque requête prise
       isolément ne voit que 12,5 tokens/s : batcher dégrade bien la latence par requête ;
    3. le débit système de ce même batch vaut 50 tokens/s, soit 2,5 fois celui de la requête
       seule : c'est le gain que l'on cherche en batchant ;
    4. le débit système agrège des requêtes de longueurs différentes (30 + 50 + 20 = 100
       tokens en 4 s font 25 tokens/s) et reste un rapport : doubler tokens et temps ensemble
       ne le change pas.

    API à faire émerger (la roadmap dit seulement « metrics », cible proposée :
    `src/inference_lab/metrics/throughput.py`)
    -------------------------------------------------------------------------
        def request_tokens_per_second(generated_tokens: int, elapsed_seconds: float) -> float: ...
        def system_tokens_per_second(
            tokens_per_request: Sequence[int], wall_clock_seconds: float
        ) -> float: ...

    Indice : deux divisions, mais deux dénominateurs différents — le temps de la requête pour
    la première, le temps MUR de l'ensemble pour la seconde (`sum(tokens_per_request)` au
    numérateur, jamais une moyenne des débits individuels : la moyenne de débits n'est pas un
    débit). Les deux fonctions doivent refuser un temps nul. Piège de fond : ne divise pas la
    somme des tokens par la somme des temps de chaque requête, ce qui reviendrait à ignorer le
    parallélisme et te redonnerait exactement le débit d'une seule requête.
    """

    pytest.skip("Roadmap TDD 13.3 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.metrics.throughput import (
        request_tokens_per_second,
        system_tokens_per_second,
    )

    # Arrange — que des nombres, aucun modèle. Décrire deux scénarios aux durées injectées :
    #           `single_tokens` = 100 tokens générés en `single_seconds` = 5,0 s de temps mur
    #           pour une requête seule ; puis un batch de quatre requêtes identiques,
    #           `batch_tokens_per_request`, une séquence de quatre fois 100 tokens, terminé en
    #           `batch_seconds` = 8,0 s de temps mur. Prévoir aussi
    #           `mixed_tokens_per_request`, trois requêtes de longueurs différentes totalisant
    #           100 tokens (30, 50 et 20), terminées en `mixed_seconds` = 4,0 s.

    # Act — demander à l'API le débit de la requête seule, le débit d'UNE requête du batch, le
    #       débit système du batch et celui du lot hétérogène. Stocker les résultats dans
    #       `single_throughput`, `batched_request_throughput`, `system_throughput` et
    #       `mixed_system_throughput`.

    # Assert 1 — référence à batch 1 : les deux notions de débit coïncident
    assert single_throughput == pytest.approx(20.0, rel=1e-12)
    assert system_tokens_per_second([100], 5.0) == pytest.approx(20.0, rel=1e-12)

    # Assert 2 — batcher dégrade la latence : une requête du batch ne voit que 12,5 tokens/s
    assert batched_request_throughput == pytest.approx(12.5, rel=1e-12)
    assert batched_request_throughput < single_throughput

    # Assert 3 — mais le débit système du batch de 4 vaut 2,5 fois celui de la requête seule
    assert system_throughput == pytest.approx(50.0, rel=1e-12)
    assert system_throughput == pytest.approx(2.5 * single_throughput, rel=1e-12)
    assert system_throughput > single_throughput

    # Assert 4 — le débit système agrège des longueurs différentes et reste un rapport
    assert mixed_system_throughput == pytest.approx(25.0, rel=1e-12)
    assert system_tokens_per_second([200, 200, 200, 200], 16.0) == pytest.approx(50.0, rel=1e-12)
