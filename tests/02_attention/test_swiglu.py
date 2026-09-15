"""Section 2.12 — SwiGLU : le MLP à porte des transformers modernes.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/02_attention/test_swiglu.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
((1, 4, 16) puis (1, 4, 8), 384 paramètres, porte nulle -> sortie nulle) : c'est volontaire. Un
test doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Dimensions constantes de la section : batch=1, seq=4, hidden=8, intermediate=16 (les têtes
n'interviennent pas ici). Le MLP représente environ deux tiers des paramètres d'un bloc : c'est
lui qui domine la mémoire des poids, alors que l'attention domine la mémoire du KV cache.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_swiglu_uses_gate_and_up_projections():
    """Roadmap 2.12 — une porte activée module multiplicativement une seconde projection.

    Objectif d'apprentissage
    ------------------------
    Un MLP classique enchaîne une projection, une activation, une projection. SwiGLU en fait
    DEUX en parallèle : la porte (`gate`), passée dans SiLU, multiplie terme à terme la voie
    directe (`up`). Le réseau peut ainsi laisser passer ou éteindre chaque canal selon
    l'entrée, ce qui explique le gain de qualité à paramètres comparables. Conséquence
    concrète : trois matrices au lieu de deux, donc plus de poids à charger et à lire — le
    coût du MLP se paie en bande passante, surtout en decode.

    Schéma mental
    -------------
        x (batch=1, seq=4, hidden=8)
            --W_gate (8, 16)-->  gate (1, 4, 16)  --SiLU-->  silu(gate)
            --W_up   (8, 16)-->  up   (1, 4, 16)
        activation = silu(gate) * up  ->  (1, 4, 16)

        SiLU(z) = z * sigmoid(z), donc SiLU(0) = 0  ->  porte nulle = canal éteint
        silu(gate) * up != silu(up) * gate : les deux voies ne sont PAS interchangeables

    Ce que ce test vérifie
    ----------------------
    1. l'activation vit dans la dimension intermédiaire (1, 4, 16), distincte de hidden=8 ;
    2. elle vaut exactement `silu(gate) * up`, avec SiLU appliqué à la SEULE porte ;
    3. une porte nulle éteint complètement la sortie (SiLU(0) = 0) ;
    4. échanger les rôles de `gate` et `up` change le résultat.

    API à faire émerger (cible roadmap : `src/inference_lab/nn/mlp/swiglu.py`)
    ------------------------------------------------------------------------
        def swiglu_activation(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor: ...

    Indice : `torch.nn.functional.silu` sert d'oracle ; SiLU s'appelle aussi « swish », d'où
    le nom. Pièges : utiliser `torch.sigmoid` seul (il manque le facteur z), appliquer
    l'activation aux deux voies, ou utiliser GELU — proche visuellement, mais numériquement
    différent de la référence Qwen.
    """

    pytest.skip("Roadmap TDD 2.12 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.mlp.swiglu import swiglu_activation

    # Arrange — batch=1, seq=4, hidden=8, intermediate=16.
    #           `gate` : tenseur (1, 4, 16) en `torch.float32`, déterministe (seed fixée),
    #           contenant des valeurs négatives ET positives.
    #           `up` : tenseur (1, 4, 16), déterministe, différent de `gate`, sans valeur nulle.
    #           `gate_zeros` : tenseur (1, 4, 16) entièrement nul.

    # Act — calculer `act = swiglu_activation(gate, up)`, `act_swapped` en échangeant les deux
    #       arguments, et `act_zero_gate = swiglu_activation(gate_zeros, up)`.

    # Assert 1 — la dimension intermédiaire (16) n'est pas la dimension cachée (8)
    assert act.shape == (1, 4, 16)
    assert swiglu_activation(gate, up).shape == (1, 4, 16)
    assert act.shape[-1] != 8

    # Assert 2 — la formule exacte : SiLU sur la porte uniquement
    torch.testing.assert_close(act, torch.nn.functional.silu(gate) * up, atol=1e-6, rtol=1e-6)

    # Assert 3 — SiLU(0) = 0 : une porte fermée éteint tous les canaux
    torch.testing.assert_close(act_zero_gate, torch.zeros(1, 4, 16), atol=1e-6, rtol=0.0)

    # Assert 4 — la porte et la voie directe ne sont pas symétriques
    assert not torch.allclose(act, act_swapped)


@pytest.mark.tdd
def test_swiglu_projects_back_to_hidden_dimension():
    """Roadmap 2.12 — le MLP élargit puis rétrécit, et travaille token par token.

    Objectif d'apprentissage
    ------------------------
    Le MLP projette dans un espace plus large (`intermediate`), calcule, puis revient à
    `hidden` : sans ce retour, impossible d'ajouter le résultat au résidu ni d'empiler les
    blocs. Le ratio intermediate/hidden est un choix d'architecture (Qwen2.5-0.5B :
    896 -> 4864, environ 5.4x) et il dicte à lui seul la majeure partie des paramètres du
    modèle. Deuxième propriété essentielle : le MLP est appliqué indépendamment à chaque
    token, aucune information ne circule entre positions — tout le mélange temporel vient de
    l'attention.

    Schéma mental
    -------------
        x (1, 4, 8)
          --W_gate (8, 16), W_up (8, 16)-->  activation (1, 4, 16)
          --W_down (16, 8)--------------->  sortie (1, 4, 8) = shape d'entrée

        paramètres = 3 matrices x 8 x 16 = 384 (un MLP classique n'en aurait que 2 x 8 x 16)
        W_down = 0            ->  sortie nulle
        tokens permutés en entrée -> sortie permutée de la même façon (calcul par token)

    Ce que ce test vérifie
    ----------------------
    1. la sortie revient exactement à la shape d'entrée (1, 4, 8), malgré l'espace
       intermédiaire de taille 16 ;
    2. les trois matrices totalisent 384 paramètres : SwiGLU en compte trois, pas deux ;
    3. une projection descendante nulle donne une sortie exactement nulle ;
    4. permuter les tokens en entrée permute la sortie de la même manière : aucun mélange
       entre positions.

    API à faire émerger (cible roadmap : `src/inference_lab/nn/mlp/swiglu.py`)
    ------------------------------------------------------------------------
        def swiglu(
            x: torch.Tensor,
            w_gate: torch.Tensor,
            w_up: torch.Tensor,
            w_down: torch.Tensor,
        ) -> torch.Tensor: ...

    Indice : `swiglu` = `swiglu_activation(x @ w_gate, x @ w_up) @ w_down` avec la convention
    `x @ w` de la section 2.1. Piège : intervertir `w_gate` et `w_down` passe parfois la
    vérification de shape si les tailles se ressemblent — pas ici, 16 != 8, et c'est précisément
    pour cela que la dimension intermédiaire est choisie différente de hidden.
    """

    pytest.skip("Roadmap TDD 2.12 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.mlp.swiglu import swiglu

    # Arrange — batch=1, seq=4, hidden=8, intermediate=16.
    #           `x` : tenseur (1, 4, 8) en `torch.float32`, déterministe (seed fixée), dont les
    #           4 tokens ont des contenus différents.
    #           `w_gate`, `w_up` : deux matrices de poids (8, 16) distinctes, déterministes.
    #           `w_down` : une matrice de poids (16, 8), déterministe et non nulle.
    #           `w_down_zeros` : une matrice (16, 8) entièrement nulle.

    # Act — calculer `out = swiglu(x, w_gate, w_up, w_down)`, `out_zero_down` avec
    #       `w_down_zeros`, et `out_flipped` en appelant `swiglu` sur `x.flip(1)` (les mêmes
    #       tokens dans l'ordre inverse) avec les poids d'origine.

    # Assert 1 — élargir puis rétrécir : on revient à la shape d'entrée
    assert out.shape == (1, 4, 8)
    assert out.shape == x.shape

    # Assert 2 — trois matrices de poids : 3 x 8 x 16 = 384 paramètres
    assert w_gate.numel() + w_up.numel() + w_down.numel() == 384

    # Assert 3 — la projection descendante conditionne toute la sortie
    torch.testing.assert_close(out_zero_down, torch.zeros(1, 4, 8), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        swiglu(x, w_gate, w_up, w_down_zeros), torch.zeros(1, 4, 8), atol=1e-6, rtol=0.0
    )

    # Assert 4 — calcul indépendant par token : permuter l'entrée permute la sortie
    torch.testing.assert_close(out_flipped, out.flip(1), atol=1e-6, rtol=1e-6)
