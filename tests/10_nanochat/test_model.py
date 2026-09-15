"""Section 10.2 — le modèle nanochat-like complet : de la pile de blocs modernes aux logits.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/10_nanochat/test_model.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
((1, 6, 32), 2 blocs, (32, 16)) : c'est volontaire. Un test doit énoncer la vérité
attendue, pas la recalculer avec la même formule que le code testé.

Modèle jouet de la section 10, constant dans tout le chapitre : `vocab_size=32`,
`hidden_size=16`, `num_layers=2`, `num_heads=4`, `num_kv_heads=2`, `head_dim=4`,
`intermediate_size=32`, `seq=6`, `batch=1`. Ce fichier assemble les briques MODERNES déjà
spécifiées séparément dans la section 2 : RMSNorm (2.9), RoPE (2.10), GQA (2.11), SwiGLU
(2.12). La structure ressemble à celle de la section 3, mais aucun composant n'est le même :
là où 3.3 empilait un bloc jouet (LayerNorm, MHA, MLP GELU), on empile ici le bloc
canonique de nanochat, Llama et Qwen2. C'est ce modèle-là, et pas le jouet de la section 3,
qui servira de terrain d'essai au KV cache (10.5) et de répétition avant le vrai Qwen
(section 12).

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
def test_nanochat_like_model_outputs_expected_logits_shape():
    """Roadmap 10.2 — une architecture moderne complète, mais la même sortie : des logits.

    Objectif d'apprentissage
    ------------------------
    RMSNorm, RoPE, GQA et SwiGLU changent tout à l'intérieur du bloc, et rien à l'extérieur :
    le contrat du modèle reste `(batch, seq)` d'ids en entrée, `(batch, seq, vocab_size)` de
    logits en sortie. C'est cette stabilité de l'interface qui rend le reste du projet
    réutilisable — la boucle de génération, l'échantillonnage, le KV cache et les benchmarks
    ne connaissent du modèle que cette signature.

    Deux propriétés spécifiques à cette architecture se lisent pourtant déjà de l'extérieur.
    La profondeur reste une donnée de config (`len(model.blocks) == num_layers`), donc le KV
    cache aura un slot par couche. Et surtout, aucune projection ne porte de biais : les LLM
    récents les ont tous supprimés (gain nul en qualité, un tenseur de moins à charger et à
    lire par couche). Un `bias` qui traîne dans `named_parameters()` est le signe qu'on a
    recopié un `torch.nn.Linear` par défaut sans réfléchir — et cela suffit à faire échouer
    le chargement des poids Qwen en section 12.

    Schéma mental
    -------------
        token_ids (batch=1, seq=6)          ids entiers dans [0, 32)
             |
             v  embedding (32, 16)
        h0 (1, 6, 16)
             |
        blocs[0] : x + GQA(RMSNorm(x))  puis  x + SwiGLU(RMSNorm(x))     num_layers=2
        blocs[1] : idem, poids distincts                                 shape INCHANGÉE
             |                                  4 têtes Q / 2 têtes KV, head_dim=4
             v                                  RoPE appliqué à Q et K, jamais à V
        norm finale (RMSNorm) -> lm_head (16 -> 32)
             |
             v
        logits (1, 6, 32)   float32, non normalisés

        softmax(logits, dim=-1).sum(dim=-1) == 1 pour chacune des 1 x 6 positions

    Ce que ce test vérifie
    ----------------------
    1. les logits ont exactement la shape `(batch, seq, vocab_size) = (1, 6, 32)`, en float32,
       et batch/seq sont ceux de l'entrée : le modèle ne réduit ni ne réordonne les positions ;
    2. la profondeur vient de la config : `model.blocks` est une `ModuleList` de 2 blocs
       distincts, avec des poids propres à chacun ;
    3. les logits sont finis et non normalisés — leur softmax somme à 1 par position, ce qui
       n'est pas le cas des logits eux-mêmes ;
    4. l'architecture est bien celle d'un LLM moderne vue de l'extérieur : le `lm_head`
       projette `hidden = 16` vers `vocab = 32` (matrice `(32, 16)`) et AUCUN paramètre du
       modèle ne s'appelle `bias`.

    API à faire émerger (cible roadmap : `src/inference_lab/models/nanochat_like/model.py`)
    --------------------------------------------------------------------------------------
        class NanochatLikeModel(torch.nn.Module):
            def __init__(self, config: NanochatLikeConfig) -> None: ...
            config: NanochatLikeConfig
            blocks: torch.nn.ModuleList
            lm_head: torch.nn.Linear      # Linear(hidden_size, vocab_size, bias=False)
            def forward(self, token_ids: torch.Tensor) -> torch.Tensor: ...

    Indice : réutilise `rms_norm` (`inference_lab.nn.normalization.rmsnorm`), `apply_rope`
    (`inference_lab.nn.positional.rope`), `grouped_query_attention`
    (`inference_lab.nn.attention.gqa`) et `swiglu` (`inference_lab.nn.mlp.swiglu`) — la
    section 10 assemble, elle ne réinvente pas. Pièges : `bias=False` sur TOUTES les
    projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`,
    `lm_head`) ; K et V se projettent en `num_kv_heads * head_dim = 8` et non en `hidden` ;
    RoPE s'applique à Q et K seulement ; la norme est APPLIQUÉE AVANT la sous-couche
    (pré-normalisation) et le résidu part de `x`, pas de `norm(x)`.
    """

    pytest.skip("Roadmap TDD 10.2 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.models.nanochat_like.model import NanochatLikeModel

    # Arrange — fixer la graine (`torch.manual_seed(0)`) puis construire :
    #           - `config`, la `NanochatLikeConfig` de la section 10.1 : vocab_size=32,
    #             hidden_size=16, num_layers=2, num_heads=4, num_kv_heads=2, head_dim=4,
    #             intermediate_size=32 ;
    #           - `model`, le `NanochatLikeModel` correspondant, en mode évaluation et sans
    #             dropout ;
    #           - `token_ids`, un tenseur `torch.long` de shape (batch=1, seq=6) dont tous les
    #             ids sont dans [0, 32).

    # Act — appeler le forward du modèle sur `token_ids` pour obtenir `logits`.

    # Assert 1 — une ligne de scores par position, une colonne par entrée du vocabulaire
    assert isinstance(model, NanochatLikeModel)
    assert logits.shape == (1, 6, 32)
    assert logits.dtype is torch.float32
    assert logits.shape[:2] == token_ids.shape

    # Assert 2 — autant de blocs que num_layers, chacun avec ses propres poids
    assert isinstance(model.blocks, torch.nn.ModuleList)
    assert len(model.blocks) == 2
    assert len(model.blocks) == model.config.num_layers
    assert model.blocks[0] is not model.blocks[1]

    # Assert 3 — des logits finis et non normalisés
    assert bool(torch.isfinite(logits).all())
    torch.testing.assert_close(
        logits.softmax(dim=-1).sum(dim=-1), torch.ones(1, 6), rtol=1e-5, atol=1e-6
    )

    # Assert 4 — signature d'un LLM moderne : projection finale hidden -> vocab, aucun biais
    assert model.lm_head.weight.shape == (32, 16)
    assert all(not name.endswith("bias") for name, _ in model.named_parameters())
