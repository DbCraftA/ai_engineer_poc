"""Section 12.6 — charger les poids réels de Qwen2.5-0.5B dans nos modules.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/12_qwen/test_weight_loading.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur (291 clés,
12 par couche, shapes (896, 896), (128, 896), (4864, 896), (151936, 896)) : c'est volontaire. Un
test doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Contrairement aux tests 12.2 à 12.5, celui-ci a besoin du VRAI checkpoint
`Qwen/Qwen2.5-0.5B-Instruct` : c'est tout son intérêt, on vérifie des noms et des shapes réels.
D'où les markers `hf` et `slow`, et un chargement en float32 sur CPU (environ 2 Go de RAM pour
494 M paramètres) pour rester déterministe et indépendant du GPU.

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
@pytest.mark.slow
def test_qwen_weights_can_be_loaded_into_internal_modules():
    """Roadmap 12.6 — un chargement de poids réussi, c'est zéro clé manquante ou inattendue.

    Objectif d'apprentissage
    ------------------------
    Un modèle n'est rien sans ses poids, et le chargement est l'étape où l'on échoue le plus
    discrètement : un nom de couche différent, une matrice transposée, un biais absent, un dtype
    non converti. La seule défense fiable est `load_state_dict(..., strict=True)` : il refuse à
    la fois les clés manquantes (notre module attend un tenseur que le checkpoint ne fournit
    pas) et les clés inattendues (le checkpoint fournit un tenseur que notre module ignore
    silencieusement). Un chargement `strict=False` qui « marche » est le meilleur moyen
    d'obtenir un modèle à moitié aléatoire qui génère du texte à peine incohérent.

    Nommer nos sous-modules comme Hugging Face (`q_proj`, `k_proj`, `v_proj`, `o_proj`,
    `gate_proj`, `up_proj`, `down_proj`, `input_layernorm`, `post_attention_layernorm`) n'est pas
    de la paresse : c'est ce qui rend le `state_dict` directement réutilisable, sans table de
    correspondance à maintenir.

    Schéma mental
    -------------
        Qwen/Qwen2.5-0.5B-Instruct : 24 couches, hidden=896, 14 têtes de query, 2 têtes KV,
        head_dim = 896 // 14 = 64, intermediate=4864, vocab=151936

        state_dict = 24 x 12 + 3 = 291 tenseurs

        par couche (12 tenseurs) :
            model.layers.N.self_attn.q_proj.weight  (896, 896)   + .bias  (896,)
            model.layers.N.self_attn.k_proj.weight  (128, 896)   + .bias  (128,)   2 x 64 = 128
            model.layers.N.self_attn.v_proj.weight  (128, 896)   + .bias  (128,)
            model.layers.N.self_attn.o_proj.weight  (896, 896)   (aucun biais)
            model.layers.N.mlp.gate_proj.weight    (4864, 896)
            model.layers.N.mlp.up_proj.weight      (4864, 896)
            model.layers.N.mlp.down_proj.weight     (896, 4864)
            model.layers.N.input_layernorm.weight         (896,)
            model.layers.N.post_attention_layernorm.weight (896,)

        hors couches (3 tenseurs) :
            model.embed_tokens.weight (151936, 896)
            model.norm.weight (896,)
            lm_head.weight (151936, 896)  <- lié à embed_tokens (tie_word_embeddings=True)

    Ce que ce test vérifie
    ----------------------
    1. les noms : 291 clés, 12 par couche, la couche 0 comme la couche 23, et aucun biais sur
       `o_proj` ni sur les projections du MLP ;
    2. les shapes réelles du checkpoint, y compris les projections K/V étroites (128 = 2 x 64)
       qui font tout l'intérêt de GQA, et la tête de sortie liée aux embeddings ;
    3. les dtypes : après un chargement demandé en float32, plus aucun tenseur n'est en bfloat16 ;
    4. la liste de clés que notre modèle attend est EXACTEMENT celle du checkpoint, le chargement
       en `strict=True` passe sans clé manquante ni inattendue, et un `state_dict` amputé d'une
       clé est refusé au lieu d'être chargé à moitié.

    API à faire émerger (cible roadmap : `src/inference_lab/models/qwen/weight_loader.py`)
    ------------------------------------------------------------------------------------
        def expected_state_dict_keys(config: QwenConfig) -> list[str]: ...
        def load_qwen_weights(
            model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]
        ) -> None: ...

    Indice : `AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)` puis
    `.state_dict()` donne la référence ; `load_qwen_weights` peut se contenter de déléguer à
    `model.load_state_dict(state_dict, strict=True)`, qui lève une `RuntimeError` détaillant les
    clés fautives. Pièges : le checkpoint est distribué en bfloat16, donc sans `dtype=` explicite
    les asserts de dtype tombent ; et comme `tie_word_embeddings` vaut True, `lm_head.weight` et
    `model.embed_tokens.weight` sont le même tenseur — ne les compte pas deux fois dans la
    mémoire du modèle.
    """

    pytest.skip("Roadmap TDD 12.6 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.models.qwen.config import from_hf_config
    from inference_lab.models.qwen.model import QwenForCausalLM
    from inference_lab.models.qwen.weight_loader import expected_state_dict_keys, load_qwen_weights

    # Arrange — cette fois le VRAI checkpoint est nécessaire
    #           (`from transformers import AutoModelForCausalLM`).
    #           `torch.manual_seed(0)` avant toute construction, tout reste sur CPU.
    #           `hf_model` : `Qwen/Qwen2.5-0.5B-Instruct` chargé avec `dtype=torch.float32`, en
    #           mode `eval()` (environ 2 Go de RAM).
    #           `hf_weights` : `hf_model.state_dict()`.
    #           `config` : notre configuration, obtenue avec `from_hf_config(hf_model.config)`.
    #           `our_model` : un `QwenForCausalLM(config)` fraîchement construit, poids encore
    #           aléatoires.
    #           `layer0_keys` : les clés de `hf_weights` qui commencent par
    #           "model.layers.0." (préfixe exact, sinon la couche 10 serait comptée aussi).
    #           `truncated_weights` : une copie de `hf_weights` privée de la clé
    #           "model.layers.0.self_attn.q_proj.weight".

    # Act — appeler `load_qwen_weights(our_model, hf_weights)`, puis relever
    #       `our_keys = sorted(our_model.state_dict())`.

    # Assert 1 — les noms, couche par couche : 24 x 12 + 3 = 291 tenseurs
    assert len(hf_weights) == 291
    assert len(layer0_keys) == 12
    assert "model.layers.0.self_attn.q_proj.weight" in hf_weights
    assert "model.layers.0.self_attn.q_proj.bias" in hf_weights
    assert "model.layers.23.mlp.down_proj.weight" in hf_weights
    assert "model.layers.0.post_attention_layernorm.weight" in hf_weights
    assert "model.norm.weight" in hf_weights
    assert "model.layers.0.self_attn.o_proj.bias" not in hf_weights
    assert "model.layers.0.mlp.gate_proj.bias" not in hf_weights

    # Assert 2 — les shapes réelles : hidden=896, head_dim=64, 2 têtes KV -> 128
    assert tuple(hf_weights["model.layers.0.self_attn.q_proj.weight"].shape) == (896, 896)
    assert tuple(hf_weights["model.layers.0.self_attn.q_proj.bias"].shape) == (896,)
    assert tuple(hf_weights["model.layers.0.self_attn.k_proj.weight"].shape) == (128, 896)
    assert tuple(hf_weights["model.layers.0.self_attn.o_proj.weight"].shape) == (896, 896)
    assert tuple(hf_weights["model.layers.0.mlp.gate_proj.weight"].shape) == (4864, 896)
    assert tuple(hf_weights["model.layers.0.mlp.down_proj.weight"].shape) == (896, 4864)
    assert tuple(hf_weights["model.embed_tokens.weight"].shape) == (151936, 896)
    assert hf_weights["lm_head.weight"].shape == hf_weights["model.embed_tokens.weight"].shape
    assert from_hf_config(hf_model.config).head_dim == 64

    # Assert 3 — tout est bien passé en float32 au chargement
    assert hf_weights["model.layers.0.self_attn.q_proj.weight"].dtype is torch.float32
    assert all(weight.dtype is torch.float32 for weight in hf_weights.values())
    assert our_model.state_dict()["model.layers.0.self_attn.q_proj.weight"].dtype is torch.float32

    # Assert 4 — strict=True : aucune clé manquante, aucune inattendue, et rien de partiel
    assert our_keys == sorted(hf_weights)
    assert sorted(expected_state_dict_keys(config)) == sorted(hf_weights)
    assert torch.equal(
        our_model.state_dict()["model.layers.0.self_attn.q_proj.weight"],
        hf_weights["model.layers.0.self_attn.q_proj.weight"],
    )
    with pytest.raises(RuntimeError):
        load_qwen_weights(QwenForCausalLM(config), truncated_weights)
