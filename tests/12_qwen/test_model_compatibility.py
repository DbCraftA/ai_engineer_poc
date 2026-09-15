"""Section 12.7 — notre Qwen complet doit reproduire les logits de la référence.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/12_qwen/test_model_compatibility.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont la shape
(1, 5, 151936), la tolérance fp32 relâchée à rtol=1e-4 / atol=1e-5 et l'égalité EXACTE des
argmax : c'est volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la
même formule que le code testé.

C'est le test final de la section : il a besoin du vrai checkpoint
`Qwen/Qwen2.5-0.5B-Instruct` (markers `hf` et `slow`), chargé en float32 sur CPU. Les tests
12.2 à 12.5 ont validé chaque brique isolément ; celui-ci valide leur assemblage sur 24
couches, c'est-à-dire l'ordre des opérations : pré-normalisation, résidus, RoPE, GQA, MLP,
norme finale, tête de sortie liée aux embeddings.

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
def test_internal_qwen_model_matches_reference_logits_on_small_input():
    """Roadmap 12.7 — mêmes poids, mêmes logits, donc même token prédit.

    Objectif d'apprentissage
    ------------------------
    Deux critères, et ils ne mesurent pas la même chose. Les logits comparés avec une tolérance
    disent « notre arithmétique est la bonne » ; l'argmax identique dit « notre modèle prend les
    mêmes décisions », ce qui est le seul critère qui compte pour un utilisateur. Un moteur
    d'inférence peut se permettre 1e-4 d'écart sur des logits d'amplitude ~20, jamais un token
    différent.

    Pourquoi 1e-4 / 1e-5 ici alors que les briques isolées tenaient à 1e-6 : l'erreur d'arrondi
    s'accumule sur 24 couches, chacune enchaînant deux normalisations, quatre projections
    d'attention et trois projections de MLP de dimension 896 ou 4864. Même en float32, l'ordre
    des sommations n'est pas le même que celui de Hugging Face (fusions, `nn.Linear` contre
    `x @ w`, chemins d'attention différents), et un écart relatif de ~1e-7 par matmul devient
    ~1e-5 en fin de pile. Exiger 1e-6 rendrait le test rouge sans qu'aucun bug n'existe ;
    accepter 1e-2 laisserait passer une convention de RoPE inversée.

    Schéma mental
    -------------
        input_ids (batch=1, seq=5), identifiants < 151936

            embed_tokens (151936, 896)  ->  h (1, 5, 896)
            24 x [ RMSNorm -> GQA (14 têtes Q, 2 têtes KV, head_dim=64) -> résidu
                   RMSNorm -> SwiGLU (896 -> 4864 -> 896)               -> résidu ]
            RMSNorm finale -> lm_head (lié à embed_tokens)
            logits (1, 5, 151936)

        logits[0, -1] -> argmax -> le token qui serait généré en greedy decoding

    Ce que ce test vérifie
    ----------------------
    1. notre modèle est bien construit depuis la configuration traduite (24 couches) et ses
       logits ont la shape et le dtype de la référence : (1, 5, 151936) en float32 ;
    2. l'égalité numérique avec la référence à rtol=1e-4 / atol=1e-5, tolérance justifiée par
       l'accumulation sur 24 couches ;
    3. la décision est identique : même argmax à chaque position, donc même token prédit à la
       dernière position (le critère qui compte vraiment) ;
    4. le modèle reste causal sur le vrai checkpoint : les logits des 3 premiers tokens ne
       changent pas quand on donne 5 tokens au lieu de 3 — c'est le prérequis du KV cache.

    API à faire émerger (cible roadmap : « modèle Qwen », cible proposée :
    `src/inference_lab/models/qwen/model.py`)
    ----------------------------------------
        class QwenForCausalLM(torch.nn.Module):
            def __init__(self, config: QwenConfig) -> None: ...

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor: ...  # logits

    Indice : assemble les modules déjà validés (`rms_norm` 12.2, `apply_rope` 12.3,
    `QwenAttention` 12.4, `swiglu` 12.5) et charge les poids avec `load_qwen_weights` (12.6).
    Mets les deux modèles en `eval()` et calcule sous `torch.no_grad()`. Pièges classiques quand
    seul cet assert casse alors que 12.2 à 12.5 sont verts : pré-norm contre post-norm (chez
    Qwen la normalisation est DANS la branche, le résidu n'est jamais normalisé), oubli de la
    `model.norm` finale avant `lm_head`, et positions repartant de 0 à chaque couche au lieu
    d'être partagées. Compare d'abord la sortie de la couche 0 si la comparaison finale
    échoue : c'est plus rapide que de chercher dans 24 couches.
    """

    pytest.skip("Roadmap TDD 12.7 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.models.qwen.config import from_hf_config
    from inference_lab.models.qwen.model import QwenForCausalLM
    from inference_lab.models.qwen.weight_loader import load_qwen_weights

    # Arrange — le VRAI checkpoint est nécessaire (`from transformers import
    #           AutoModelForCausalLM, AutoTokenizer`).
    #           `torch.manual_seed(0)` avant toute construction, tout reste sur CPU.
    #           `hf_model` : `Qwen/Qwen2.5-0.5B-Instruct` chargé avec `dtype=torch.float32`, en
    #           mode `eval()`.
    #           `input_ids` : tenseur d'entiers de shape (1, 5), identifiants VALIDES
    #           (strictement inférieurs à 151936), déterministes — par exemple la tokenisation
    #           d'un prompt court avec `AutoTokenizer.from_pretrained`.
    #           `our_model` : un `QwenForCausalLM(from_hf_config(hf_model.config))` en mode
    #           `eval()`, dont les poids ont été chargés depuis `hf_model.state_dict()` avec
    #           `load_qwen_weights`.

    # Act — sous `torch.no_grad()`, calculer la référence `logits_ref` (attribut `.logits` de la
    #       sortie de `hf_model(input_ids=input_ids)`) et nos `logits = our_model(input_ids)` ;
    #       calculer aussi `logits_prefix`, nos logits sur les 3 premiers tokens seulement
    #       (`input_ids[:, :3]`).

    # Assert 1 — même contrat de sortie que la référence
    assert isinstance(our_model, QwenForCausalLM)
    assert from_hf_config(hf_model.config).num_hidden_layers == 24
    assert load_qwen_weights(our_model, hf_model.state_dict()) is None
    assert logits.shape == (1, 5, 151936)
    assert logits.shape == logits_ref.shape
    assert logits.dtype is torch.float32
    assert bool(logits.isfinite().all())

    # Assert 2 — égalité numérique, tolérance justifiée par 24 couches d'accumulation
    torch.testing.assert_close(logits, logits_ref, rtol=1e-4, atol=1e-5)

    # Assert 3 — même décision : argmax identique partout, donc même token prédit
    assert torch.equal(logits.argmax(dim=-1), logits_ref.argmax(dim=-1))
    assert int(logits[0, -1].argmax()) == int(logits_ref[0, -1].argmax())

    # Assert 4 — causalité sur le vrai modèle : le passé ne dépend pas du futur
    assert logits_prefix.shape == (1, 3, 151936)
    torch.testing.assert_close(logits_prefix, logits_ref[:, :3], rtol=1e-4, atol=1e-5)
