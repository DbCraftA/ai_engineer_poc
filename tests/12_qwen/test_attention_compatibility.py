"""Section 12.4 — notre attention GQA doit reproduire `Qwen2Attention`.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/12_qwen/test_attention_compatibility.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont les shapes de poids de
Qwen2 (16, 16) et (8, 16), les 7 clés de son `state_dict`, et la sortie de la référence à
rtol=1e-5 / atol=1e-6 : c'est volontaire. Un test doit énoncer la vérité attendue, pas la
recalculer avec la même formule que le code testé.

Dimensions : batch=1, seq=4, hidden=16, num_heads=4, head_dim=4, num_kv_heads=2. Aucun
checkpoint n'est téléchargé : un `Qwen2Attention` isolé, construit sur une `Qwen2Config`
réduite avec des poids aléatoires seedés, est une référence exacte. Ce test agrège tout ce
qu'ont installé les sections 2.1 à 2.10 : projections Q/K/V, découpage en têtes, RoPE, masque
causal, répétition des têtes KV, projection de sortie.

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
def test_our_gqa_matches_qwen_reference_on_small_input():
    """Roadmap 12.4 — biais sur Q, K, V, aucun biais sur la projection de sortie.

    Objectif d'apprentissage
    ------------------------
    C'est le bloc où toutes les briques de la section 2 doivent s'assembler dans le bon ordre :
    projeter, découper en têtes, tourner Q et K avec la RoPE, masquer le futur, répéter les
    têtes KV pour couvrir les têtes de query, pondérer V, recoller les têtes, projeter en
    sortie. Une seule inversion et la sortie reste plausible mais fausse.

    La particularité d'architecture à connaître pour Qwen2 : contrairement à Llama, ses
    projections `q_proj`, `k_proj` et `v_proj` ont un BIAIS, alors que `o_proj` n'en a pas.
    Si notre module oublie les trois biais, le chargement des poids en 12.6 signalera des clés
    inattendues ; s'il ajoute un biais à `o_proj`, c'est une clé manquante. Les deux erreurs
    sont silencieuses si l'on se contente de comparer des shapes de sortie.

    Schéma mental
    -------------
        x (batch=1, seq=4, hidden=16)

            q_proj (16, 16) + biais (16,)  ->  Q (1, 4 têtes,  4, head_dim=4)
            k_proj  (8, 16) + biais  (8,)  ->  K (1, 2 têtes KV, 4, 4)
            v_proj  (8, 16) + biais  (8,)  ->  V (1, 2 têtes KV, 4, 4)
            o_proj (16, 16) SANS biais     ->  sortie (1, 4, 16)

        RoPE sur Q et K -> masque causal -> K/V répétés 4 // 2 = 2 fois -> softmax -> V
        échelle des scores : 1 / sqrt(head_dim) = 1 / 2 = 0.5

        Qwen2.5-0.5B : 14 têtes de query, 2 têtes KV, head_dim=64 -> cache divisé par 7

    Ce que ce test vérifie
    ----------------------
    1. la structure des poids est celle de Qwen2 : 7 tenseurs, biais sur Q/K/V, pas sur
       `o_proj`, et des projections K/V deux fois plus étroites que Q (GQA) ;
    2. le `state_dict` de la référence se charge tel quel dans notre module en `strict=True` :
       aucune clé manquante ni inattendue ;
    3. à poids identiques, notre sortie est celle de `Qwen2Attention` à rtol=1e-5 / atol=1e-6 ;
    4. l'attention est causale : la sortie du premier token est la même que l'on donne 1 token
       ou 4 tokens en entrée — c'est la propriété qui autorisera le KV cache.

    API à faire émerger (cible roadmap : « Qwen attention », cible proposée :
    `src/inference_lab/models/qwen/attention.py`)
    --------------------------------------------
        class QwenAttention(torch.nn.Module):
            def __init__(self, config: QwenConfig) -> None: ...

            # sous-modules nommés comme chez Hugging Face, pour charger ses poids tels quels :
            # q_proj, k_proj, v_proj : torch.nn.Linear(..., bias=True)
            # o_proj                 : torch.nn.Linear(..., bias=False)

            def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
                ...

    Indice : réutilise `apply_rope` (2.10), `repeat_kv_heads` (2.8), `split_heads` / `merge_heads`
    (2.7) et `apply_causal_mask` (2.4) plutôt que de tout réécrire. Côté référence, deux pièges :
    `Qwen2Attention` doit être construit avec `attn_implementation="eager"` dans la config, sinon
    l'implémentation d'attention n'est pas résolue ; et appelé avec `attention_mask=None` il n'est
    PAS causal — il faut lui passer un masque additif de shape (1, 1, 4, 4) valant 0.0 sur les
    positions autorisées et `-inf` au-dessus de la diagonale. Sa signature est
    `forward(hidden_states, position_embeddings=(cos, sin), attention_mask=...)` et elle renvoie
    un couple `(sortie, poids d'attention)`.
    """

    pytest.skip("Roadmap TDD 12.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.models.qwen.attention import QwenAttention
    from inference_lab.models.qwen.config import from_hf_config

    # Arrange — référence Hugging Face locale, sans checkpoint (`from transformers import
    #           Qwen2Config` et `from transformers.models.qwen2.modeling_qwen2 import
    #           Qwen2Attention, Qwen2RotaryEmbedding`).
    #           `hf_config` : `Qwen2Config` réduite (hidden_size=16, num_hidden_layers=2,
    #           num_attention_heads=4, num_key_value_heads=2, intermediate_size=32,
    #           vocab_size=32, rope_theta=1000000.0) construite avec
    #           `attn_implementation="eager"`.
    #           `hf_attn` : un `Qwen2Attention(hf_config, layer_idx=0)` en mode `eval()`, poids
    #           aléatoires mais déterministes (`torch.manual_seed(0)` avant construction).
    #           `hf_weights` : son `state_dict()`, la source de vérité des poids.
    #           `our_attn` : un `QwenAttention(from_hf_config(hf_config))`.
    #           `x` : tenseur (1, 4, 16) en `torch.float32`, déterministe.
    #           `positions` : tenseur d'entiers 1D valant 0, 1, 2, 3.
    #           `causal_mask_hf` : masque additif (1, 1, 4, 4) valant 0.0 sur le triangle
    #           inférieur (diagonale incluse) et `-inf` ailleurs, pour l'appel Hugging Face.

    # Act — charger `hf_weights` dans `our_attn` avec `load_state_dict(..., strict=True)` et
    #       nommer le résultat `report` ; puis, sous `torch.no_grad()`, calculer la référence
    #       `out_hf` (premier élément du couple renvoyé par `hf_attn`, avec les `cos, sin` de
    #       `Qwen2RotaryEmbedding(hf_config)` et `causal_mask_hf`) et notre sortie
    #       `out = our_attn(x, positions)` ; calculer enfin `out_first_token`, notre sortie sur
    #       le seul premier token `x[:, :1]` avec `positions[:1]`.

    # Assert 1 — structure de poids propre à Qwen2 : biais sur Q/K/V, pas sur o_proj
    assert isinstance(our_attn, QwenAttention)
    assert len(hf_weights) == 7
    assert tuple(hf_weights["q_proj.weight"].shape) == (16, 16)
    assert tuple(hf_weights["q_proj.bias"].shape) == (16,)
    assert tuple(hf_weights["k_proj.weight"].shape) == (8, 16)
    assert tuple(hf_weights["v_proj.weight"].shape) == (8, 16)
    assert "o_proj.bias" not in hf_weights
    assert our_attn.o_proj.bias is None
    assert our_attn.q_proj.bias is not None

    # Assert 2 — les poids de la référence entrent dans notre module sans adaptation
    assert list(report.missing_keys) == []
    assert list(report.unexpected_keys) == []
    assert sorted(our_attn.state_dict()) == sorted(hf_weights)
    assert sorted(QwenAttention(from_hf_config(hf_config)).state_dict()) == sorted(hf_weights)

    # Assert 3 — à poids identiques, mêmes sorties : notre attention EST celle de Qwen2
    assert out.shape == (1, 4, 16)
    assert out.dtype is torch.float32
    torch.testing.assert_close(out, out_hf, rtol=1e-5, atol=1e-6)

    # Assert 4 — causalité : le premier token ignore les 3 suivants
    torch.testing.assert_close(out_first_token, out[:, :1], rtol=1e-5, atol=1e-6)
