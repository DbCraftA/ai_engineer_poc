"""Section 7.4 — benchmark de prefill : latence et débit d'entrée en tokens/s.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/07_performance/test_prefill_benchmark.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(prompt de 16 tokens, 4 appels du forward) : c'est volontaire. Un test doit énoncer la
vérité attendue, pas la recalculer avec la même formule que le code testé.

RÈGLE D'OR de la section 7 : aucune durée absolue n'est assertée, elle dépendrait de la
machine. Sur une métrique dérivée comme le débit, on asserte la COHÉRENCE INTERNE — ici
`input_tokens_per_second == prompt_tokens / latency_seconds` — plus le signe et le
comptage d'appels d'une fonction jouet instrumentée. Ce test s'appuie sur le protocole de
mesure de 7.1 / 7.2 (warmup hors chronomètre, médiane de plusieurs répétitions).

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
@pytest.mark.perf
def test_prefill_benchmark_reports_latency_and_input_tokens_per_second():
    """Roadmap 7.4 — le prefill se mesure en latence ET en tokens d'entrée par seconde.

    Objectif d'apprentissage
    ------------------------
    Le prefill est la première phase de l'inférence : un seul forward traite les N tokens du
    prompt d'un coup et remplit le KV cache (sections 4.2 à 4.4). Il est *compute-bound* :
    les GEMM travaillent sur des matrices (N, hidden), donc le GPU est bien occupé et le coût
    croît avec la longueur du prompt.

    Deux métriques différentes en décrivent la performance :

        latency_seconds         -> temps d'attente vu par l'utilisateur avant le 1er token
                                   (le fameux TTFT, time to first token) ;
        input_tokens_per_second -> capacité d'ingestion du moteur, comparable entre deux
                                   longueurs de prompt et entre deux machines.

    Publier l'une sans l'autre est trompeur : un prompt deux fois plus long double la
    latence sans changer le débit. Le débit est la métrique à comparer, la latence celle
    qui compte pour l'utilisateur.

    Schéma mental
    -------------
        prompt_ids (batch=1, seq=16) --forward jouet--> logits, KV cache rempli

        warmup=1, repeats=3  ->  1 + 3 = 4 appels du forward, 3 mesures agrégées
        latency_seconds       = médiane des 3 mesures (valeur inconnue, jamais assertée)
        input_tokens_per_second = 16 / latency_seconds     <- relation exacte, assertée

    Ce que ce test vérifie
    ----------------------
    1. le nombre de tokens d'entrée est lu sur l'axe séquence du prompt : 16 tokens,
       ids entiers de shape (1, 16) ;
    2. le protocole de 7.1 est respecté : 1 warmup + 3 répétitions = 4 appels du forward ;
    3. la latence rapportée est un `float` strictement positif, et le débit aussi (aucune
       borne absolue : on ne teste pas la vitesse de la machine) ;
    4. la relation exacte entre les deux métriques :
       `input_tokens_per_second == prompt_tokens / latency_seconds`.

    API à faire émerger (cible roadmap « benchmark », cible proposée :
    `src/inference_lab/benchmarks/prefill.py`)
    -----------------------------------------------------------------
        @dataclass(frozen=True)
        class PrefillResult:
            prompt_tokens: int
            latency_seconds: float
            input_tokens_per_second: float

        def benchmark_prefill(
            forward: Callable[[torch.Tensor], object],
            prompt_ids: torch.Tensor,
            *,
            warmup: int = 1,
            repeats: int = 3,
        ) -> PrefillResult: ...

    Indice : réutilise `run_benchmark` de 7.1 en lui passant `lambda: forward(prompt_ids)`,
    prends `run.median_seconds` comme latence, et lis le nombre de tokens sur
    `prompt_ids.shape[-1]` (pas `numel()`, qui compterait le batch). Pièges : ne divise pas
    par la somme des durées (tu obtiendrais un débit `repeats` fois trop petit) et n'oublie
    pas `torch.cuda.synchronize()` avant de lire l'horloge dès qu'un vrai modèle GPU est
    mesuré, sinon tu chronomètres un lancement de kernel asynchrone (section 6.2).
    """

    pytest.skip("Roadmap TDD 7.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.benchmarks.prefill import benchmark_prefill

    # Arrange — pas de vrai modèle : un forward jouet déterministe et instrumenté. Construire
    #           `prompt_ids`, un tenseur d'ids de tokens en `torch.long` de shape (1, 16) ;
    #           `calls`, une liste vide servant de compteur d'appels ; et `toy_forward`, une
    #           fonction qui prend un tenseur d'ids, enregistre son appel dans `calls` et
    #           renvoie un vrai résultat calculé à partir des ids (une petite couche
    #           d'embedding suffit) pour que la durée mesurée soit non nulle.

    # Act — lancer le benchmark de prefill sur `toy_forward` et `prompt_ids` avec 1 itération
    #       de warmup et 3 répétitions mesurées, puis garder le résultat dans `result`.

    # Assert 1 — le nombre de tokens d'entrée vient de l'axe séquence du prompt
    assert prompt_ids.shape == (1, 16)
    assert prompt_ids.dtype is torch.long
    assert result.prompt_tokens == 16

    # Assert 2 — warmup hors mesure : 1 + 3 = 4 appels du forward
    assert len(calls) == 4

    # Assert 3 — les deux métriques sont des flottants strictement positifs
    assert isinstance(result.latency_seconds, float)
    assert result.latency_seconds > 0.0
    assert result.input_tokens_per_second > 0.0

    # Assert 4 — cohérence interne exacte : débit = tokens d'entrée / latence
    assert result.input_tokens_per_second == pytest.approx(16 / result.latency_seconds, rel=1e-12)
