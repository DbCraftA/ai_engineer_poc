"""Section 12.5 — notre MLP SwiGLU doit reproduire `Qwen2MLP`.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/12_qwen/test_mlp_compatibility.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont les shapes de poids de
Qwen2 ((32, 16) et (16, 32)), les 1536 paramètres du MLP réduit, et la sortie de la référence
à rtol=1e-5 / atol=1e-6 : c'est volontaire. Un test doit énoncer la vérité attendue, pas la
recalculer avec la même formule que le code testé.

Dimensions : batch=1, seq=4, hidden=16, intermediate=32. Aucun checkpoint n'est téléchargé :
un `Qwen2MLP` isolé sur une `Qwen2Config` réduite, poids aléatoires seedés, est une référence
exacte. Le MLP pèse environ deux tiers des paramètres d'un bloc (Qwen2.5-0.5B : 896 -> 4864),
c'est donc lui qui domine la lecture des poids en decode.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
@pytest.mark.model
@pytest.mark.hf
def test_our_mlp_matches_qwen_reference_on_small_input():
    """Roadmap 12.5 — `down(silu(gate(x)) * up(x))`, sans aucun biais.

    Objectif d'apprentissage
    ------------------------
    Le MLP de Qwen2 est un SwiGLU en trois matrices, toutes sans biais : la porte `gate_proj`
    passe dans SiLU et module multiplicativement `up_proj`, puis `down_proj` ramène vers
    `hidden`. Deux erreurs sont possibles et parfaitement silencieuses : intervertir `gate_proj`
    et `up_proj` (la formule n'est pas symétrique, SiLU ne s'applique qu'à la porte), et oublier
    que `torch.nn.Linear` stocke sa matrice en (out_features, in_features) alors que la
    convention `x @ w` du dépôt attend (in_features, out_features). Charger les poids Hugging
    Face sans transposer donne parfois des shapes compatibles — ici (32, 16) contre (16, 32) —
    et un résultat faux.

    Schéma mental
    -------------
        x (batch=1, seq=4, hidden=16)

        Hugging Face (nn.Linear, calcule x @ w.T) :
            gate_proj.weight (32, 16), up_proj.weight (32, 16), down_proj.weight (16, 32)
        notre convention (x @ w) :
            w_gate = gate_proj.weight.T (16, 32), w_up (16, 32), w_down (32, 16)

        gate = x @ w_gate --SiLU--> silu(gate)      (1, 4, 32)
        up   = x @ w_up                             (1, 4, 32)
        sortie = (silu(gate) * up) @ w_down         (1, 4, 16)

        paramètres = 3 x 16 x 32 = 1536, aucun biais
        Qwen2.5-0.5B : 3 x 896 x 4864 ~ 13.1 M paramètres par couche

    Ce que ce test vérifie
    ----------------------
    1. la structure de poids de `Qwen2MLP` : 3 tenseurs, aucun biais, 1536 paramètres, et des
       shapes transposées par rapport à notre convention ;
    2. à poids identiques, notre sortie est celle de `Qwen2MLP` à rtol=1e-5 / atol=1e-6, et
       revient bien à hidden=16 depuis l'espace intermédiaire de 32 ;
    3. la porte et la voie directe ne sont pas interchangeables : échanger `gate` et `up` donne
       un résultat différent de la référence ;
    4. le MLP travaille token par token : le calcul sur le seul token 2 donne la même ligne que
       dans la sortie complète.

    API à faire émerger (cible roadmap : « Qwen MLP », cible proposée : réutilisation de
    `src/inference_lab/nn/mlp/swiglu.py` de la section 2.12)
    -------------------------------------------------------
        def swiglu(
            x: torch.Tensor,
            w_gate: torch.Tensor,
            w_up: torch.Tensor,
            w_down: torch.Tensor,
        ) -> torch.Tensor: ...

    Indice : si la section 2.12 est verte, il n'y a rien à écrire — seulement à transposer les
    trois poids Hugging Face au moment de les passer à `swiglu`. Oracle de l'activation :
    `torch.nn.functional.silu` ; c'est exactement l'`act_fn` de `Qwen2MLP`. Piège : GELU
    ressemble à SiLU sur une courbe mais s'en écarte de plus de 1e-2, la tolérance de ce test
    le détecte immédiatement.
    """

    pytest.skip("Roadmap TDD 12.5 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.mlp.swiglu import swiglu

    # Arrange — référence Hugging Face locale, sans checkpoint (`from transformers import
    #           Qwen2Config` et `from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP`).
    #           `hf_config` : `Qwen2Config` réduite (hidden_size=16, num_hidden_layers=2,
    #           num_attention_heads=4, num_key_value_heads=2, intermediate_size=32, vocab_size=32).
    #           `hf_mlp` : un `Qwen2MLP(hf_config)` en mode `eval()`, poids déterministes
    #           (`torch.manual_seed(0)` avant construction).
    #           `hf_weights` : son `state_dict()`.
    #           `w_gate`, `w_up`, `w_down` : les trois poids de `hf_mlp` TRANSPOSÉS pour notre
    #           convention `x @ w`, soit (16, 32), (16, 32) et (32, 16).
    #           `x` : tenseur (1, 4, 16) en `torch.float32`, déterministe, valeurs positives ET
    #           négatives (sinon SiLU n'est jamais exercé sur sa partie négative).

    # Act — sous `torch.no_grad()`, calculer la référence `out_hf = hf_mlp(x)` et notre
    #       `out = swiglu(x, w_gate, w_up, w_down)` ; calculer aussi `out_swapped` en échangeant
    #       `w_gate` et `w_up`, et `out_token` sur le seul token `x[:, 2:3]`.

    # Assert 1 — trois matrices, aucun biais, et la transposition à ne pas oublier
    assert len(hf_weights) == 3
    assert "gate_proj.bias" not in hf_weights
    assert "down_proj.bias" not in hf_weights
    assert tuple(hf_weights["gate_proj.weight"].shape) == (32, 16)
    assert tuple(hf_weights["down_proj.weight"].shape) == (16, 32)
    assert tuple(w_gate.shape) == (16, 32)
    assert tuple(w_down.shape) == (32, 16)
    assert sum(weight.numel() for weight in hf_weights.values()) == 1536

    # Assert 2 — à poids identiques, mêmes sorties, et retour à hidden=16
    assert out.shape == (1, 4, 16)
    assert out.dtype is torch.float32
    torch.testing.assert_close(out, out_hf, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(swiglu(x, w_gate, w_up, w_down), out_hf, rtol=1e-5, atol=1e-6)

    # Assert 3 — SiLU ne s'applique qu'à la porte : les deux voies ne sont pas symétriques
    assert not torch.allclose(out_swapped, out_hf, rtol=1e-3, atol=1e-3)

    # Assert 4 — aucun mélange entre positions : un token se calcule seul
    torch.testing.assert_close(out_token, out_hf[:, 2:3], rtol=1e-5, atol=1e-6)
