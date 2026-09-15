"""Section 9.5 — SwiGLU en Triton : fusionner la porte et la voie directe en un passage.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/09_triton/test_swiglu.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(shape (1, 4, 4864), porte nulle -> sortie nulle, tolérances rtol=1e-5 / atol=1e-6) : c'est
volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule que
le code testé.

Cap du chapitre : le kernel répond à une question mémoire, pas à un souhait de vitesse. Ici la
question est la fusion : en PyTorch, `silu(gate) * up` matérialise un tenseur intermédiaire de
la taille de la dimension intermédiaire (4864 par token chez Qwen2.5-0.5B), écrit puis relu en
mémoire globale pour rien. Un kernel fusionné lit `gate` et `up`, écrit la sortie, et c'est
tout. Le test ne mesure aucune performance : il vérifie la CORRECTION contre
`swiglu_activation`, la référence PyTorch de la section 2.12, qui sert d'oracle — même nom,
mêmes arguments, même formule. La section 2.12 doit donc être verte avant celle-ci.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
@pytest.mark.triton
def test_triton_swiglu_matches_pytorch_reference():
    """Roadmap 9.5 — fusionner SiLU et le produit ne doit rien changer au résultat.

    Objectif d'apprentissage
    ------------------------
    Le MLP concentre environ deux tiers des paramètres d'un bloc, et son activation est
    parfaitement élémentaire : chaque sortie ne dépend que de `gate[i]` et `up[i]`. Aucune
    réduction, aucune communication entre voies — c'est le cas d'école de la fusion. Le kernel
    devient un simple `charge, charge, calcule, écris`, et le trafic mémoire tombe de deux
    aller-retours à un seul. C'est aussi la limite claire du chapitre : le gain est en
    passages mémoire, alors que la sortie doit rester, elle, celle de la référence PyTorch.

    Point numérique : contrairement à l'addition de 9.2, `silu(z) = z * sigmoid(z)` fait
    intervenir une exponentielle. `tl.sigmoid` et `torch.nn.functional.silu` n'utilisent pas
    forcément la même approximation matérielle de `exp`, et les derniers bits peuvent différer :
    `torch.equal` est donc à exclure, même sans réduction. La tolérance reste très serrée
    (rtol=1e-5, atol=1e-6) parce qu'aucune somme ne vient accumuler l'erreur — plus serrée que
    pour le RMSNorm de 9.4.

    Schéma mental
    -------------
        gate (batch=1, seq=4, intermediate=4864), up (1, 4, 4864)   -- Qwen2.5-0.5B
        numel = 4 x 4864 = 19456 éléments  ->  BLOCK = 256  ->  76 programs

        PyTorch  : t = silu(gate) (écrit 19456 valeurs), puis t * up (relit tout)
        Triton   : charge gate, charge up, écrit silu(gate) * up  -> un seul passage

        SiLU(0) = 0 * sigmoid(0) = 0        ->  porte nulle = canal éteint, sortie 0.0 exacte
        silu(gate) * up != silu(up) * gate  ->  les deux voies ne sont pas interchangeables

    Ce que ce test vérifie
    ----------------------
    1. la fusion écrit un seul tenseur, de shape (1, 4, 4864) : on est dans la dimension
       intermédiaire, distincte de la dimension cachée 896, en float32 et sur `"cuda"` ;
    2. le résultat correspond à `swiglu_activation` de la section 2.12, l'oracle, avec
       rtol=1e-5 et atol=1e-6 : la sigmoïde interdit l'égalité bit à bit, mais rien n'autorise
       une dérive plus large ;
    3. une porte entièrement nulle éteint la sortie exactement (SiLU(0) = 0), et échanger les
       rôles de `gate` et `up` change le résultat : l'activation ne porte que sur la porte ;
    4. avec un nombre d'éléments non multiple de la taille de bloc, la sortie reste conforme à
       l'oracle : le masquage ne laisse ni trou ni valeur non initialisée.

    API à faire émerger (cible roadmap : `src/inference_lab/kernels/triton/swiglu.py`)
    -------------------------------------------------------------------------------
        def triton_swiglu_activation(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor: ...

    Indice : la référence à reproduire est `torch.nn.functional.silu(gate) * up` (section
    2.12) ; côté kernel, `g * tl.sigmoid(g) * u` sur des offsets linéaires, avec le même motif
    `mask = offs < n_elements` que 9.2 puisque l'opération est élémentaire. Pièges : appliquer
    SiLU aux deux voies, remplacer SiLU par GELU (visuellement proche, numériquement différent
    de la référence Qwen), et supposer que les deux entrées ont le même `stride` sans exiger
    qu'elles soient contiguës.
    """

    pytest.skip("Roadmap TDD 9.5 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.kernels.triton.swiglu import triton_swiglu_activation
    from inference_lab.nn.mlp.swiglu import swiglu_activation

    # Arrange — batch=1, seq=4, intermediate=4864 (la dimension intermédiaire de
    #           Qwen2.5-0.5B). Appeler `torch.manual_seed(0)` d'abord, puis créer, tous en
    #           `torch.float32`, contigus et alloués directement sur `"cuda"` : `gate` de shape
    #           (1, 4, 4864), valeurs de magnitude proche de 1, négatives ET positives ; `up` de
    #           shape (1, 4, 4864), différent de `gate` et sans valeur nulle ; `gate_zeros` de
    #           shape (1, 4, 4864) entièrement nul ; `gate_tail` et `up_tail` de shape (1, 1000),
    #           dont le nombre d'éléments n'est PAS un multiple de 256 (1000 = 3 x 256 + 232).

    # Act — calculer `act = triton_swiglu_activation(gate, up)`, la référence PyTorch
    #       `reference = swiglu_activation(gate, up)`, `act_swapped` en échangeant les deux
    #       arguments du kernel, `act_zero_gate = triton_swiglu_activation(gate_zeros, up)` et
    #       `act_tail = triton_swiglu_activation(gate_tail, up_tail)`.

    # Assert 1 — un seul tenseur écrit, dans la dimension intermédiaire
    assert act.shape == (1, 4, 4864)
    assert triton_swiglu_activation(gate, up).shape == (1, 4, 4864)
    assert act.shape[-1] != 896
    assert act.dtype is torch.float32
    assert act.device.type == "cuda"

    # Assert 2 — l'oracle est la référence PyTorch de la section 2.12, tolérance serrée
    torch.testing.assert_close(act, reference, rtol=1e-5, atol=1e-6)

    # Assert 3 — SiLU(0) = 0, et la porte n'est pas interchangeable avec la voie directe
    assert torch.equal(act_zero_gate, torch.zeros(1, 4, 4864, device="cuda"))
    assert not torch.allclose(act, act_swapped)

    # Assert 4 — masquage : une taille non multiple du bloc reste conforme à l'oracle
    torch.testing.assert_close(
        act_tail, swiglu_activation(gate_tail, up_tail), rtol=1e-5, atol=1e-6
    )
