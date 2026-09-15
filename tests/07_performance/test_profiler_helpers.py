"""Section 7.6 — profiling : savoir OÙ passe le temps, opération par opération.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/07_performance/test_profiler_helpers.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(shape (4, 8), présence d'une opération `matmul`/`addmm`) : c'est volontaire. Un test doit
énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

RÈGLE D'OR de la section 7 : on n'asserte jamais une durée. Ici on vérifie que le profiler
CAPTURE les bonnes opérations et qu'il ne fausse pas le résultat numérique ; les temps
relevés ne sont pas comparés à des seuils. Le profiling est limité à
`ProfilerActivity.CPU` pour que ce test reste exécutable sans GPU.

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
def test_profiler_can_capture_model_operations():
    """Roadmap 7.6 — un profil dit OÙ passe le temps, là où un benchmark ne dit que combien.

    Objectif d'apprentissage
    ------------------------
    Les benchmarks de 7.1 à 7.5 disent COMBIEN de temps prend une passe ; ils ne disent pas
    OÙ il part. Le profiler, lui, décompose l'exécution en opérations ATen (`aten::addmm`,
    `aten::matmul`, `aten::softmax`, `aten::copy_`, ...) et permet de répondre aux vraies
    questions d'optimisation : les GEMM dominent-ils (compute-bound) ou est-ce une avalanche
    de petites opérations élémentaires (overhead de lancement, candidat à la fusion des
    sections 8 et 9) ? Un `aten::copy_` inattendu trahit souvent un transfert CPU/GPU ou un
    `.contiguous()` caché.

    `torch.profiler.profile` prend les activités à observer : `ProfilerActivity.CPU` pour les
    opérateurs côté hôte, `ProfilerActivity.CUDA` pour les kernels GPU. On se limite ici au
    CPU pour que le test tourne partout, y compris en CI sans carte.

    Schéma mental
    -------------
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            output = linear(x)          # x (4, 8) --Linear(8, 8)--> output (4, 8)

        prof.key_averages() -> une entrée par opération, agrégée :
            aten::linear, aten::t, aten::transpose, aten::addmm, aten::expand, ...
                                                    ^^^^^^^^^^^
                        le GEMM de la couche linéaire : c'est lui qu'on cherche

    Ce que ce test vérifie
    ----------------------
    1. le profil est non vide et chaque entrée porte un nom d'opération non vide, préfixé
       `aten::` (les opérateurs du dispatcher PyTorch) ;
    2. l'opération de multiplication matricielle attendue est bien capturée : au moins une
       entrée contient `matmul` ou `addmm` ;
    3. le profiling est transparent pour le calcul : la sortie profilée a la shape (4, 8) et
       vaut numériquement la sortie non profilée ;
    4. le rapport lisible par un humain est disponible sous forme de texte non vide.

    API à faire émerger (cible roadmap : `src/inference_lab/profiling/pytorch_profiler.py`)
    ---------------------------------------------------------------------------------------
        @dataclass(frozen=True)
        class ProfileReport:
            operation_names: list[str]
            table: str

        def profile_operations(
            fn: Callable[[], torch.Tensor],
        ) -> tuple[torch.Tensor, ProfileReport]: ...

    Indice : `from torch.profiler import ProfilerActivity, profile`, puis
    `with profile(activities=[ProfilerActivity.CPU]) as prof: output = fn()`. Les noms
    s'obtiennent avec `[event.key for event in prof.key_averages()]` et le texte avec
    `prof.key_averages().table()`. Pièges : ne lis rien sur `prof` avant la sortie du bloc
    `with` (le profil n'est pas encore consolidé), n'utilise pas `record_shapes=True` par
    réflexe (les noms restent les mêmes mais le coût monte), et n'ajoute
    `ProfilerActivity.CUDA` que sous garde `torch.cuda.is_available()`.
    """

    pytest.skip("Roadmap TDD 7.6 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.profiling.pytorch_profiler import profile_operations

    # Arrange — un calcul jouet représentatif d'un bloc de LLM, déterministe (seed fixée) :
    #           `layer`, une `torch.nn.Linear` de 8 entrées vers 8 sorties en float32, et `x`,
    #           un tenseur d'entrée `torch.float32` de shape (4, 8). Calculer d'abord
    #           `reference`, la sortie de `layer` sur `x` SANS profiling, qui servira d'oracle.

    # Act — profiler l'appel de `layer` sur `x` avec le helper, en récupérant la sortie dans
    #       `output` et le rapport dans `report`.

    # Assert 1 — le profil contient des opérations nommées du dispatcher PyTorch
    assert len(report.operation_names) >= 1
    assert all(isinstance(name, str) and name for name in report.operation_names)
    assert any(name.startswith("aten::") for name in report.operation_names)

    # Assert 2 — le GEMM de la couche linéaire est bien capturé
    assert any("matmul" in name or "addmm" in name for name in report.operation_names)

    # Assert 3 — profiler ne change pas le résultat du calcul
    assert output.shape == (4, 8)
    torch.testing.assert_close(output, reference, rtol=1e-6, atol=1e-6)

    # Assert 4 — un rapport texte exploitable est disponible
    assert isinstance(report.table, str)
    assert len(report.table) > 0
