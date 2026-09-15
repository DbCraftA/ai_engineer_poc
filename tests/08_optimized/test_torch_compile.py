"""Section 8.3 — `torch.compile` accélère l'exécution sans changer les sorties.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/08_optimized/test_torch_compile.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
((1, 4, 8), (1, 6, 8), tolérance 1e-5/1e-6) : c'est volontaire. Un test doit énoncer la
vérité attendue, pas la recalculer avec la même formule que le code testé.

Ce test est un test de NON-RÉGRESSION, pas un benchmark : la vitesse se mesure dans la
section 7, ici on ne prouve qu'une chose, la seule qui autorise à compiler en production —
la compilation ne modifie pas les nombres produits par le modèle. Il tourne sur CPU, sur un
module minuscule (hidden=8, intermediate=16), et reste donc exécutable partout.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_compiled_model_matches_eager_output():
    """Roadmap 8.3 — compiler change le chemin d'exécution, jamais le résultat.

    Objectif d'apprentissage
    ------------------------
    `torch.compile` capture le graphe Python du `forward` (TorchDynamo), le réécrit et le
    confie à un backend qui fusionne les opérations élément par élément et supprime des
    allers-retours en mémoire globale. Le mode eager, lui, exécute une opération à la fois,
    en relisant les tenseurs à chaque étape. Le gain vient donc de la même arithmétique
    exécutée autrement — c'est le pendant, au niveau du modèle entier, de la fusion manuelle
    de la section 8.4. Pour un moteur d'inférence, cela n'a de valeur que si la sortie est
    inchangée : un test comme celui-ci est ce qui permet d'activer la compilation sans
    revalider la qualité du modèle. Deux propriétés à retenir : le PREMIER appel paie la
    compilation (des secondes, pas des microsecondes), et changer la shape d'entrée déclenche
    une recompilation — d'où la séparation prefill / decode et les shapes fixes du KV cache.

    Schéma mental
    -------------
        x (batch=1, seq=4, hidden=8)
            --Linear(8, 16)--> (1, 4, 16) --SiLU--> (1, 4, 16) --Linear(16, 8)--> (1, 4, 8)

        model(x)                     -> eager_out    (1, 4, 8), float32
        compile_model(model)(x)      -> compiled_out (1, 4, 8), float32   [1er appel : lent]
        compile_model(model)(x)      -> même valeur                       [2e appel : cache]

        x_longer (1, 6, 8) -> nouvelle shape -> recompilation -> mêmes valeurs que l'eager

    Ce que ce test vérifie
    ----------------------
    1. la compilation renvoie un NOUVEL objet appelable, encore un `torch.nn.Module`, qui
       partage les mêmes objets paramètres que le modèle d'origine (aucune copie de poids) ;
    2. la sortie compilée est égale à la sortie eager, shape (1, 4, 8) et tolérance fp32
       rtol=1e-5 / atol=1e-6 ;
    3. le deuxième appel, servi par le graphe déjà compilé, donne la même valeur que le
       premier, et le modèle eager d'origine produit toujours le même résultat qu'avant ;
    4. une entrée de longueur de séquence différente (1, 6, 8) reste correcte : la
       recompilation ne change pas les valeurs.

    API à faire émerger (cible roadmap : « compilation helper », cible proposée
    `src/inference_lab/models/compile.py`)
    -------------------------------------------------------------------------
        def compile_model(model: torch.nn.Module, backend: str = "inductor") -> torch.nn.Module: ...

    Indice : le corps tient en `torch.compile(model, backend=backend)`. Le test passe
    explicitement `backend="aot_eager"` : il capture et rejoue un vrai graphe, mais sans
    générer de code machine, donc il ne dépend d'aucune toolchain C/Triton et reste vert sur
    CPU (le backend par défaut `inductor` est celui de la production). Pièges : compiler dans
    un `with torch.no_grad():` et en `model.eval()` pour comparer deux inférences, pas deux
    entraînements ; et ne rien conclure d'un chronomètre sur le premier appel, qui inclut la
    compilation.
    """

    pytest.skip("Roadmap TDD 8.3 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.models.compile import compile_model

    # Arrange — `model` : un `torch.nn.Module` minuscule et déterministe (seed fixée), en
    #           `.eval()`, sur CPU : `Linear(8, 16)` -> `SiLU` -> `Linear(16, 8)`, donc
    #           hidden=8 et intermediate=16, sans dropout ni aucune source d'aléa au forward.
    #           `x` : un tenseur (1, 4, 8) en `torch.float32` (batch=1, seq=4, hidden=8),
    #           valeurs déterministes non constantes.
    #           `x_longer` : un tenseur (1, 6, 8), même dtype, seule la longueur change.

    # Act — calculer `eager_out = model(x)`, puis `compiled = compile_model(model,
    #       backend="aot_eager")` et `compiled_out = compiled(x)` (ce premier appel déclenche
    #       la compilation, il est lent), puis `compiled_out_again = compiled(x)`,
    #       `eager_out_after = model(x)` et enfin `compiled_out_longer = compiled(x_longer)`
    #       avec `eager_out_longer = model(x_longer)`. Tout sous `torch.no_grad()`.

    # Assert 1 — un nouvel objet, toujours un module, sans duplication des poids
    assert compiled is not model
    assert isinstance(compile_model(model, backend="aot_eager"), torch.nn.Module)
    assert all(a is b for a, b in zip(compiled.parameters(), model.parameters(), strict=True))

    # Assert 2 — même sortie que le mode eager, à la tolérance fp32
    assert eager_out.shape == (1, 4, 8)
    assert compiled_out.shape == (1, 4, 8)
    torch.testing.assert_close(compiled_out, eager_out, rtol=1e-5, atol=1e-6)

    # Assert 3 — le graphe compilé est stable et n'a pas altéré le modèle d'origine
    torch.testing.assert_close(compiled_out_again, compiled_out, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(eager_out_after, eager_out, rtol=1e-5, atol=1e-6)

    # Assert 4 — nouvelle longueur de séquence : recompilation, mêmes valeurs
    assert compiled_out_longer.shape == (1, 6, 8)
    torch.testing.assert_close(compiled_out_longer, eager_out_longer, rtol=1e-5, atol=1e-6)
