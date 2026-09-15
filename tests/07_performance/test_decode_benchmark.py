"""Section 7.5 — benchmark de decode : temps par token de sortie (TPOT).

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/07_performance/test_decode_benchmark.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(8 tokens générés, 10 appels de l'étape de decode) : c'est volontaire. Un test doit énoncer
la vérité attendue, pas la recalculer avec la même formule que le code testé.

RÈGLE D'OR de la section 7 : aucune durée absolue n'est assertée. Sur une métrique dérivée
comme le TPOT, on asserte la COHÉRENCE INTERNE — ici
`time_per_output_token == total_decode_time / generated_tokens` — le comptage d'appels
d'une étape de decode jouet, et le fait que le warmup reste hors du décompte (7.1).

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
def test_decode_benchmark_reports_time_per_output_token():
    """Roadmap 7.5 — le decode se mesure par token généré, pas par requête.

    Objectif d'apprentissage
    ------------------------
    Le decode est la seconde phase de l'inférence : une étape par token, chacune ne traitant
    qu'UNE position (seq=1) mais relisant tous les poids du modèle et tout le KV cache. Il est
    *memory-bound* : le GPU passe son temps à charger des octets, pas à calculer. Sa métrique
    naturelle est donc le temps par token de sortie :

        TPOT = total_decode_time / generated_tokens      (aussi appelé ITL, inter-token latency)

    C'est ce chiffre que ressent l'utilisateur pendant le streaming (30 ms/token = ~33 tokens/s
    affichés), et c'est lui qui bouge quand on change de dtype (FP16 vs FP32), qu'on active le
    KV cache (section 4) ou qu'on fusionne des kernels (sections 8 et 9). Le mesurer par token
    et non par requête rend deux runs de longueurs différentes comparables.

    Attention à ne pas mélanger prefill et decode dans le même chronomètre : le prefill
    (7.4) est compute-bound et son coût dépend de la longueur du prompt. Le chronomètre du
    decode démarre APRÈS le prefill, une fois le premier token disponible.

    Schéma mental
    -------------
        hidden (batch=1, seq=1, hidden=8) : une seule position par étape, c'est le decode

        warmup=2  ->  2 étapes hors chronomètre
        [ étape 1 | étape 2 | ... | étape 8 ]  <- total_decode_time, un seul chronomètre
        10 appels de l'étape jouet au total, mais 8 tokens comptés

        TPOT = total_decode_time / 8            <- relation exacte, assertée
        tokens/s = 1 / TPOT                     <- relation exacte, assertée

    Ce que ce test vérifie
    ----------------------
    1. le cadre du decode : l'étape jouet travaille sur une seule position, un tenseur
       (batch=1, seq=1, hidden=8) en float32 ;
    2. le warmup n'est pas compté : 2 + 8 = 10 appels de l'étape, mais 8 tokens générés ;
    3. le temps total et le TPOT sont des flottants strictement positifs, et le TPOT est
       strictement plus petit que le total puisque 8 tokens ont été générés ;
    4. les deux relations exactes : `TPOT == total_decode_time / generated_tokens` et
       `output_tokens_per_second == 1 / TPOT`.

    API à faire émerger (cible roadmap « benchmark », cible proposée :
    `src/inference_lab/benchmarks/decode.py`)
    -----------------------------------------------------------------
        @dataclass(frozen=True)
        class DecodeResult:
            generated_tokens: int
            total_decode_time_seconds: float
            time_per_output_token_seconds: float
            output_tokens_per_second: float

        def benchmark_decode(
            decode_step: Callable[[], object],
            *,
            generated_tokens: int,
            warmup: int = 2,
        ) -> DecodeResult: ...

    Indice : ici on chronomètre la BOUCLE entière (`start = time.perf_counter()` avant les
    `generated_tokens` étapes, une seule lecture à la fin), contrairement à 7.2 qui mesure
    chaque répétition séparément : le TPOT est une moyenne par token, robuste au coût d'une
    itération isolée. Pièges : ne compte pas les étapes de warmup dans `generated_tokens`, et
    n'oublie pas `torch.cuda.synchronize()` avant la dernière lecture d'horloge sur GPU
    (section 6.2), sans quoi tu mesures la file de lancement des kernels, pas le calcul.
    """

    pytest.skip("Roadmap TDD 7.5 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.benchmarks.decode import benchmark_decode

    # Arrange — pas de vrai modèle : une étape de decode jouet, déterministe et instrumentée.
    #           Construire `hidden`, un tenseur `torch.float32` de shape (1, 1, 8) représentant
    #           l'unique position traitée par étape ; `calls`, une liste vide servant de
    #           compteur d'appels ; et `decode_step`, une fonction sans argument qui enregistre
    #           son appel dans `calls` et fait un vrai petit calcul sur `hidden` (le résultat
    #           importe peu, mais le calcul doit avoir lieu pour que la durée soit non nulle).

    # Act — lancer le benchmark de decode sur `decode_step` avec 2 étapes de warmup et 8 tokens
    #       à générer, puis garder le résultat dans `result`.

    # Assert 1 — le decode traite une seule position par étape
    assert hidden.shape == (1, 1, 8)
    assert hidden.dtype is torch.float32

    # Assert 2 — le warmup s'exécute mais ne compte pas comme des tokens générés
    assert len(calls) == 10
    assert result.generated_tokens == 8

    # Assert 3 — durées strictement positives, et un token coûte moins que les 8 réunis
    assert isinstance(result.total_decode_time_seconds, float)
    assert result.total_decode_time_seconds > 0.0
    assert result.time_per_output_token_seconds > 0.0
    assert result.time_per_output_token_seconds < result.total_decode_time_seconds

    # Assert 4 — cohérence interne exacte : TPOT = temps total / tokens générés
    assert result.time_per_output_token_seconds == pytest.approx(
        result.total_decode_time_seconds / 8, rel=1e-12
    )
    assert result.output_tokens_per_second == pytest.approx(
        1.0 / result.time_per_output_token_seconds, rel=1e-12
    )
