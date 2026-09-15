"""Section 7.7 — résultat de benchmark : les métadonnées qui rendent une mesure reproductible.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/07_performance/test_benchmark_result.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(batch 1, prompt 128 tokens, sortie 64 tokens, dtype « torch.float32 ») : c'est
volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule
que le code testé.

RÈGLE D'OR de la section 7 : aucune durée absolue n'est assertée. Ce test-ci ne mesure même
rien : il vérifie le CONTENANT des mesures produites par 7.1 à 7.5, c'est-à-dire les
métadonnées sans lesquelles un chiffre de latence n'est ni interprétable ni comparable.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices. Même raison pour l'import
# du module cible : c'est ton code d'`Act` qui l'appellera, le linter le voit donc inutilisé.
# ruff: noqa: F401, F821

import dataclasses
import json

import pytest
import torch


@pytest.mark.tdd
@pytest.mark.perf
def test_benchmark_result_contains_hardware_and_model_metadata():
    """Roadmap 7.7 — une latence sans son contexte matériel et modèle n'est pas une mesure.

    Objectif d'apprentissage
    ------------------------
    « 42 ms par token » ne veut rien dire tout seul. La même commande donne des chiffres
    radicalement différents selon le modèle, le dtype, le GPU, le batch et les longueurs de
    séquence. Un résultat de benchmark doit donc transporter tout ce qui permet de le
    reproduire et de le comparer :

        modèle     -> quel checkpoint (Qwen2.5-0.5B n'est pas Qwen2.5-7B) ;
        dtype      -> FP32 / FP16 / BF16 change la bande passante et les unités utilisées ;
        device     -> `cpu` ou `cuda:0` ;
        nom du GPU -> une A100 80 Go et une T4 n'ont pas la même HBM ni le même TFLOPS ;
        batch      -> le débit total monte avec le batch, la latence par requête aussi ;
        longueurs  -> prompt (coût du prefill) et sortie (coût du decode) ;
        version de torch -> un changement de version peut changer les kernels choisis.

    C'est la structure qu'on sérialisera en JSON/CSV pour comparer deux runs, tracer une
    courbe latence/batch, ou vérifier une non-régression après une optimisation (sections 8
    et 9). Un `dataclass` est le bon outil : champs nommés et typés, égalité structurelle
    gratuite, conversion en dict immédiate.

    Schéma mental
    -------------
        BenchmarkResult
          |-- model_name       "Qwen/Qwen2.5-0.5B"
          |-- dtype            "torch.float32"      <- str(torch.float32), sérialisable
          |-- device           "cpu"
          |-- gpu_name         None                 <- renseigné seulement sur un run CUDA
          |-- batch_size       1
          |-- prompt_tokens    128
          |-- output_tokens    64
          |-- latency_seconds  <mesure de 7.4 / 7.5, valeur non assertée>
          |-- torch_version    torch.__version__    <- relevé par le code, pas saisi à la main

        to_dict() -> mêmes clés, valeurs JSON-sérialisables

    Ce que ce test vérifie
    ----------------------
    1. le résultat est bien un `dataclass` et expose au moins les neuf champs indispensables
       à la reproductibilité ;
    2. les valeurs de description du run sont conservées telles quelles, et le dtype est
       normalisé en texte sérialisable (`"torch.float32"`) ;
    3. les types sont ceux attendus (`int` pour les tailles, `float` pour la latence, `str`
       pour les identifiants), l'environnement est relevé automatiquement
       (`torch_version == torch.__version__`) et `gpu_name` vaut `None` sur un run CPU ;
    4. la sérialisation en dict est complète (mêmes clés que les champs) et passe par
       `json.dumps` sans erreur.

    API à faire émerger (cible roadmap : `src/inference_lab/benchmarks/result.py`)
    -----------------------------------------------------------------------------
        @dataclass(frozen=True)
        class BenchmarkResult:
            model_name: str
            dtype: str
            device: str
            gpu_name: str | None
            batch_size: int
            prompt_tokens: int
            output_tokens: int
            latency_seconds: float
            torch_version: str

            def to_dict(self) -> dict[str, object]: ...

        def build_benchmark_result(
            *,
            model_name: str,
            dtype: torch.dtype,
            device: str,
            batch_size: int,
            prompt_tokens: int,
            output_tokens: int,
            latency_seconds: float,
        ) -> BenchmarkResult: ...

    Indice : `dataclasses.asdict(self)` suffit pour `to_dict`, mais uniquement si aucun champ
    ne contient d'objet non sérialisable — c'est pour cela que le dtype est stocké via
    `str(dtype)` et non comme `torch.dtype`. La fabrique renseigne elle-même
    `torch_version=torch.__version__` et `gpu_name=torch.cuda.get_device_name(0)` quand le
    device est CUDA, `None` sinon : ces informations ne doivent JAMAIS être saisies à la main
    dans un rapport. Piège : `frozen=True` interdit de muter le résultat après coup, ce qui
    est voulu (un rapport de mesure est immuable).
    """

    pytest.skip("Roadmap TDD 7.7 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.benchmarks.result import build_benchmark_result

    # Arrange — décrire un run CPU du petit modèle de la roadmap : modèle
    #           `"Qwen/Qwen2.5-0.5B"`, dtype `torch.float32`, device `"cpu"`, batch de 1,
    #           prompt de 128 tokens, sortie de 64 tokens, et une latence en secondes
    #           strictement positive (valeur arbitraire : ce test ne mesure rien).

    # Act — construire `result` avec la fabrique, puis en tirer `payload`, sa version dict.

    # Assert 1 — c'est un dataclass et il porte les champs indispensables
    assert dataclasses.is_dataclass(result)
    assert {field.name for field in dataclasses.fields(result)} >= {
        "model_name",
        "dtype",
        "device",
        "gpu_name",
        "batch_size",
        "prompt_tokens",
        "output_tokens",
        "latency_seconds",
        "torch_version",
    }

    # Assert 2 — la description du run est conservée, le dtype normalisé en texte
    assert result.model_name == "Qwen/Qwen2.5-0.5B"
    assert result.dtype == "torch.float32"
    assert result.device == "cpu"
    assert result.batch_size == 1
    assert result.prompt_tokens == 128
    assert result.output_tokens == 64

    # Assert 3 — types attendus et environnement relevé automatiquement
    assert isinstance(result.batch_size, int)
    assert isinstance(result.prompt_tokens, int)
    assert isinstance(result.latency_seconds, float)
    assert isinstance(result.torch_version, str)
    assert result.torch_version == torch.__version__
    assert result.gpu_name is None

    # Assert 4 — sérialisation complète et compatible JSON
    assert isinstance(payload, dict)
    assert set(payload) == {field.name for field in dataclasses.fields(result)}
    assert payload["prompt_tokens"] == 128
    assert payload["gpu_name"] is None
    assert json.loads(json.dumps(payload))["dtype"] == "torch.float32"
