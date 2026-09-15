"""Section 13.5 — schéma de résultat : les clés qui rendent deux runs comparables.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/13_benchmark/test_result_metadata.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(batch 1, prompt 128 tokens, sortie 64 tokens, dtype « torch.float32 ») : c'est volontaire.
Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code
testé.

Aucune mesure ici, donc aucune durée assertée : ce fichier teste le CONTENANT des métriques de
13.1 à 13.4. Il ne redéfinit PAS de structure de résultat : le dataclass `BenchmarkResult`
existe déjà, il est spécifié par `tests/07_performance/test_benchmark_result.py` (roadmap 7.7)
dans `src/inference_lab/benchmarks/result.py`. Créer un second schéma ici garantirait deux
formats divergents dans les rapports ; la section 13 ajoute seulement, au-dessus de lui, la
sérialisation stable et le critère de comparabilité entre deux runs.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices. Même raison pour l'import
# du module cible : c'est ton code d'`Act` qui l'appellera, le linter le voit donc inutilisé.
# ruff: noqa: F401, F821

import json

import pytest


@pytest.mark.tdd
@pytest.mark.perf
def test_results_record_model_dtype_batch_context_and_hardware():
    """Roadmap 13.5 — comparer deux runs suppose que tout sauf la mesure soit identique.

    Objectif d'apprentissage
    ------------------------
    Un rapport de benchmark sert à répondre à une question du type « le passage en FP16 a-t-il
    amélioré le TPOT ? ». Cette question n'a de sens que si les deux runs comparés partagent le
    même modèle, le même batch, les mêmes longueurs de contexte et le même matériel : sinon on
    compare deux expériences différentes et l'écart observé ne s'attribue à rien. D'où deux
    exigences sur le schéma :

        clés d'identité  -> modèle, dtype, device, batch, prompt_tokens, output_tokens :
                            elles décrivent l'EXPÉRIENCE, et deux runs ne sont comparables
                            que si elles coïncident, à l'exception de celle que l'on fait
                            varier volontairement ;
        clés de mesure    -> latence et métriques dérivées : ce sont les valeurs qui DOIVENT
                            différer d'un run à l'autre, sinon on n'a rien mesuré.

    Deuxième exigence : la sérialisation doit être STABLE. Un rapport se stocke en JSON/CSV,
    se relit des semaines plus tard, se compare avec `diff` en CI pour détecter une régression
    (sections 8 et 9). Un ordre de clés qui change à chaque exécution, ou un `torch.dtype`
    non sérialisable, suffit à casser cette chaîne. `json.dumps(..., sort_keys=True)` rend la
    sortie déterministe et l'aller-retour JSON doit être l'identité.

    Schéma mental
    -------------
        BenchmarkResult (7.7)                  couche 13.5
          model_name     "Qwen/Qwen2.5-0.5B"   |  clés d'identité obligatoires :
          dtype          "torch.float32"       |    model_name, dtype, device,
          device         "cpu"                 |    batch_size, prompt_tokens, output_tokens
          gpu_name       None                  |
          batch_size     1                     |  result_to_json -> texte trié, stable
          prompt_tokens  128                   |  json.loads(texte) == to_dict()
          output_tokens  64                    |
          latency_seconds  <mesure>            |  runs_are_comparable(a, b) -> bool
          torch_version  <relevé auto>         |    ignore les clés de mesure

        run A : FP32, latence 2,0 s  |  run B : FP32, latence 1,0 s  -> comparables
        run C : FP16, latence 1,0 s                                  -> NON comparable à A

    Ce que ce test vérifie
    ----------------------
    1. la liste des clés d'identité est explicite et vaut exactement les six clés
       modèle / dtype / device / batch / prompt / sortie, et le dict d'un résultat les contient
       toutes (aucune ne manque) ;
    2. les valeurs d'identité sont conservées telles quelles (batch 1, 128 et 64 tokens) et le
       dtype est déjà normalisé en texte `"torch.float32"` par le schéma de 7.7 ;
    3. la sérialisation est stable et fidèle : deux appels donnent le même texte, les clés y
       sont triées, et l'aller-retour JSON redonne exactement le dict de départ ;
    4. le critère de comparabilité ignore les mesures mais pas l'identité : deux runs de
       latences différentes mais de même configuration sont comparables, un run en FP16 ne
       l'est pas ; et une clé d'identité manquante est signalée nommément.

    API à faire émerger (la roadmap dit « benchmark schema », cible proposée :
    `src/inference_lab/metrics/schema.py`, au-dessus du `BenchmarkResult` de
    `src/inference_lab/benchmarks/result.py` défini en 7.7)
    -------------------------------------------------------------------------
        RUN_IDENTITY_KEYS: frozenset[str]

        def result_to_json(result: BenchmarkResult) -> str: ...
        def missing_identity_keys(payload: Mapping[str, object]) -> set[str]: ...
        def runs_are_comparable(left: BenchmarkResult, right: BenchmarkResult) -> bool: ...

    Indice : `result_to_json` s'écrit `json.dumps(result.to_dict(), sort_keys=True)` — c'est
    `sort_keys=True` qui rend le texte déterministe, pas l'ordre de déclaration du dataclass.
    `runs_are_comparable` compare les projections des deux résultats sur `RUN_IDENTITY_KEYS`,
    jamais les objets entiers (`left == right` serait faux dès que la latence diffère, c'est-à-
    dire toujours). Pièges : n'ajoute pas `latency_seconds` ni `gpu_name` aux clés d'identité
    (l'un est la mesure, l'autre se déduit du device) ; et ne mets pas `torch_version` dans le
    critère de comparabilité de ce test, sinon deux runs séparés par une mise à jour de PyTorch
    deviendraient incomparables alors qu'on veut justement pouvoir enquêter sur l'écart.
    """

    pytest.skip("Roadmap TDD 13.5 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.metrics.schema import (
        RUN_IDENTITY_KEYS,
        missing_identity_keys,
        result_to_json,
        runs_are_comparable,
    )

    # Arrange — trois descriptions de run construites avec la fabrique de 7.7
    #           (`build_benchmark_result` de `inference_lab.benchmarks.result`), aucune mesure
    #           réelle : `run_fp32`, un run CPU de `"Qwen/Qwen2.5-0.5B"` en `torch.float32`,
    #           batch 1, prompt de 128 tokens, sortie de 64 tokens, latence strictement
    #           positive ; `run_fp32_faster`, identique en tout point sauf une latence
    #           différente ; `run_fp16`, identique à `run_fp32` sauf le dtype `torch.float16`.

    # Act — obtenir `payload`, le dict de `run_fp32`, et `blob`, sa sérialisation JSON par
    #       l'API du schéma.

    # Assert 1 — les clés d'identité sont explicites et toutes présentes dans le rapport
    assert RUN_IDENTITY_KEYS == frozenset(
        {
            "model_name",
            "dtype",
            "device",
            "batch_size",
            "prompt_tokens",
            "output_tokens",
        }
    )
    assert missing_identity_keys(payload) == set()

    # Assert 2 — les valeurs d'identité sont conservées, le dtype déjà normalisé en texte
    assert payload["model_name"] == "Qwen/Qwen2.5-0.5B"
    assert payload["dtype"] == "torch.float32"
    assert payload["batch_size"] == 1
    assert payload["prompt_tokens"] == 128
    assert payload["output_tokens"] == 64

    # Assert 3 — sérialisation stable : texte déterministe, clés triées, aller-retour fidèle
    assert blob == result_to_json(run_fp32)
    assert list(json.loads(blob)) == sorted(payload)
    assert json.loads(blob) == payload

    # Assert 4 — comparabilité : la mesure varie librement, l'identité non
    assert runs_are_comparable(run_fp32, run_fp32_faster) is True
    assert runs_are_comparable(run_fp32, run_fp16) is False
    assert missing_identity_keys({"model_name": "Qwen/Qwen2.5-0.5B"}) == {
        "dtype",
        "device",
        "batch_size",
        "prompt_tokens",
        "output_tokens",
    }
