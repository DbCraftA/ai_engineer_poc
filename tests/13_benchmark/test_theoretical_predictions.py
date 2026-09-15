"""Section 13.6 — prédiction théorique contre mesure : même schéma, valeurs différentes.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/13_benchmark/test_theoretical_predictions.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(1 ms/token prédit soit 1000 tokens/s, 2,5 ms/token mesuré soit 400 tokens/s, efficacité 0,4) :
c'est volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même
formule que le code testé.

RÈGLE D'OR, reprise de la section 7 : aucune durée absolue mesurée n'est assertée, et ce test
ne compare surtout PAS prédiction et mesure en valeur — elles diffèrent toujours, c'est même
tout l'intérêt. Ce qui est asserté, c'est que les deux produisent le MÊME SCHÉMA de clés, donc
qu'elles sont confrontables terme à terme ; les chiffres du test proviennent de durées
injectées.

C'est le point d'arrivée de la roadmap : la boucle « prédire -> mesurer -> expliquer l'écart »
du README. La prédiction vient de la section 5 (FLOPs, octets, roofline), la mesure des
sections 7 et 13.1 à 13.4 ; leur rapport est l'efficacité mesurée, comprise entre 0 et 1, et
c'est cet écart qui désigne ce qu'il reste à optimiser.

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
def test_theoretical_model_produces_comparable_prediction_schema_to_measurements():
    """Roadmap 13.6 — une prédiction n'est utile que si elle est confrontable à la mesure.

    Objectif d'apprentissage
    ------------------------
    La section 5 sait calculer une borne théorique du decode sans exécuter le modèle. Le decode
    étant *memory-bound* (5.10), sa limite est la bande passante : chaque token exige de relire
    tous les poids, donc

        TPOT_prédit      = octets_lus_par_token / bande_passante
        tokens/s_prédit  = 1 / TPOT_prédit

    La mesure, elle, vient de 13.2. Les deux chiffres ne coïncideront jamais : la mesure inclut
    l'overhead de lancement des kernels, les kernels non fusionnés, la bande passante réelle
    (60 à 80 % de la valeur commerciale), le KV cache lu en plus des poids, la couche Python.
    L'écart n'est donc pas un bug, c'est l'INFORMATION :

        efficacité_mesurée = mesure / prédiction        (dans ]0, 1] par construction)

        0,8 -> le moteur est proche de la limite matérielle : optimiser le code ne rapportera
               plus grand-chose, il faut changer la charge (batcher, quantifier) ;
        0,4 -> plus de la moitié de la bande passante est perdue : il y a un vrai coupable à
               trouver, et c'est le profiler de 7.6 qui le nommera ;
        > 1  -> la prédiction est fausse (octets sous-estimés) : une mesure ne dépasse pas une
               borne physique.

    Pour que cette division ait un sens, prédiction et mesure doivent parler la même langue :
    mêmes clés, mêmes unités. C'est précisément ce que ce test verrouille, et c'est pour cela
    qu'il asserte un SCHÉMA plutôt que des valeurs.

    Schéma mental
    -------------
        prédiction (section 5)                        mesure (13.2)
        ----------------------                        -------------
        poids relus   = 1e9 octets/token              200 tokens générés en 0,5 s
        bande passante = 1e12 octets/s

        time_per_output_token = 1e9 / 1e12            = 0,5 / 200
                              = 1,0e-3 s (1 ms)       = 2,5e-3 s (2,5 ms)
        tokens_per_second     = 1000,0                 = 400,0

        mêmes clés {"tokens_per_second", "time_per_output_token"}, valeurs différentes

        efficacité = 400 / 1000 = 0,4  ==  1,0e-3 / 2,5e-3 = 0,4   (les deux voies concordent)

    Ce que ce test vérifie
    ----------------------
    1. prédiction et mesure exposent exactement le même jeu de clés,
       `{"tokens_per_second", "time_per_output_token"}`, ce qui les rend confrontables terme à
       terme ;
    2. la prédiction chiffrée du roofline mémoire : 1,0e-3 s par token, donc 1000 tokens/s ;
    3. la mesure chiffrée issue des durées injectées : 2,5e-3 s par token, donc 400 tokens/s —
       différente de la prédiction, et jamais meilleure qu'elle ;
    4. l'efficacité mesurée vaut 0,4, se calcule indifféremment sur le débit ou sur le temps
       par token, et reste dans ]0, 1] ; une efficacité de 1,0 signifierait que la mesure
       atteint la borne théorique.

    API à faire émerger (la roadmap dit « calculators/benchmark », cible proposée :
    `src/inference_lab/metrics/schema.py`, module déjà visé par 13.5, qui s'appuie sur
    `src/inference_lab/calculators/roofline.py` de la section 5)
    -------------------------------------------------------------------------------
        DECODE_METRIC_KEYS: frozenset[str]   # {"tokens_per_second", "time_per_output_token"}

        def predicted_decode_metrics(
            *, bytes_read_per_token: float, bandwidth_bytes_per_s: float
        ) -> dict[str, float]: ...

        def measured_decode_metrics(
            *, generated_tokens: int, decode_seconds: float
        ) -> dict[str, float]: ...

        def measured_efficiency(
            measured: Mapping[str, float], predicted: Mapping[str, float]
        ) -> float: ...

    Indice : les deux fabriques doivent construire leur dict à partir de `DECODE_METRIC_KEYS`,
    et non répéter les littéraux de clés à deux endroits — c'est ce qui garantit que le schéma
    ne divergera pas. `measured_decode_metrics` réutilise `time_per_output_token` de 13.2 au
    lieu de refaire la division. Pièges : `measured_efficiency` est un rapport de DÉBITS
    (`measured / predicted`) ou, de façon équivalente, un rapport INVERSÉ de temps par token
    (`predicted / measured`) — se tromper de sens donne 2,5 au lieu de 0,4, une « efficacité »
    supérieure à 1 qui devrait immédiatement alerter ; et garde des octets/s en entrée, pas des
    Go/s, sinon la prédiction est fausse d'un facteur 1e9 (même piège qu'en 5.9).
    """

    pytest.skip("Roadmap TDD 13.6 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.metrics.schema import (
        DECODE_METRIC_KEYS,
        measured_decode_metrics,
        measured_efficiency,
        predicted_decode_metrics,
    )

    # Arrange — que des nombres, aucun modèle exécuté. Côté prédiction, le budget de la section
    #           5 pour Qwen2.5-0.5B en FP16 : `bytes_read_per_token` = 1e9 octets de poids relus
    #           à chaque token, sur un matériel de `bandwidth_bytes_per_s` = 1e12 octets/s
    #           (1000 Go/s, le matériel de référence de 5.9). Côté mesure, des durées injectées
    #           déjà relevées : `generated_tokens` = 200 tokens produits en `decode_seconds` =
    #           0,5 s de decode.

    # Act — demander à l'API le dictionnaire de métriques prédites, celui des métriques
    #       mesurées, puis l'efficacité de la mesure par rapport à la prédiction. Stocker les
    #       résultats dans `predicted`, `measured` et `efficiency`.

    # Assert 1 — même schéma de clés des deux côtés : les métriques sont confrontables
    assert set(predicted) == {"tokens_per_second", "time_per_output_token"}
    assert set(measured) == set(predicted)
    assert set(DECODE_METRIC_KEYS) == set(predicted)

    # Assert 2 — la prédiction du roofline mémoire, écrite en dur
    assert predicted["time_per_output_token"] == pytest.approx(1.0e-3, rel=1e-12)
    assert predicted["tokens_per_second"] == pytest.approx(1000.0, rel=1e-12)

    # Assert 3 — la mesure, différente et jamais au-dessus de la borne théorique
    assert measured["time_per_output_token"] == pytest.approx(2.5e-3, rel=1e-12)
    assert measured["tokens_per_second"] == pytest.approx(400.0, rel=1e-12)
    assert measured["tokens_per_second"] < predicted["tokens_per_second"]

    # Assert 4 — l'écart s'interprète : efficacité de 0,4, cohérente sur les deux métriques
    assert efficiency == pytest.approx(0.4, rel=1e-12)
    assert efficiency == pytest.approx(
        predicted["time_per_output_token"] / measured["time_per_output_token"], rel=1e-12
    )
    assert 0.0 < efficiency <= 1.0
