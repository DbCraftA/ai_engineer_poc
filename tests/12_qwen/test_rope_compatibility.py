"""Section 12.3 — notre RoPE doit tourner Q et K exactement comme Qwen2.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/12_qwen/test_rope_compatibility.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues viennent de la référence
Hugging Face (`Qwen2RotaryEmbedding` + `apply_rotary_pos_emb`) et les tolérances sont écrites
en dur (fp32 : rtol=1e-5, atol=1e-6) : c'est volontaire. Un test doit énoncer la vérité
attendue, pas la recalculer avec la même formule que le code testé.

Dimensions du test : batch=1, seq=4, num_heads=4, num_kv_heads=2, head_dim=4, hidden=16.
Aucun poids n'est nécessaire : la RoPE n'a pas de paramètre appris, seulement `theta` et les
positions. C'est donc le test de compatibilité le plus rapide de la section — et celui qui
attrape le bug le plus fréquent quand on réimplémente un modèle Hugging Face.

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
def test_our_rope_matches_qwen_reference():
    """Roadmap 12.3 — LE piège : Hugging Face tourne par moitiés, pas par paires voisines.

    Objectif d'apprentissage
    ------------------------
    Deux implémentations de RoPE circulent, mathématiquement équivalentes à une permutation des
    canaux près, mais NUMÉRIQUEMENT différentes sur les mêmes poids :

    - convention « moitiés » (Hugging Face, donc Qwen2, Llama) : le vecteur de tête est coupé
      en deux, `x[..., :head_dim//2]` et `x[..., head_dim//2:]`, et le canal j est apparié au
      canal j + head_dim//2 ; c'est ce que fait `rotate_half`, qui renvoie
      `cat((-x2, x1), dim=-1)` ;
    - convention « entrelacée » (article original, beaucoup de réimplémentations) : les paires
      sont les canaux voisins (x0, x1), (x2, x3), ...

    Les deux conservent la norme, les deux donnent l'identité à la position 0, les deux ont des
    scores qui ne dépendent que de l'écart de positions : AUCUNE des propriétés de la section
    2.10 ne les distingue. Seule une comparaison canal par canal avec la référence les
    sépare. Avec des poids entraînés, se tromper de convention ne produit ni exception ni NaN :
    juste un modèle qui génère du texte incohérent. C'est exactement ce que ce test empêche.

    Schéma mental
    -------------
        head_dim = 4, positions = [0, 1, 2, 3], theta = 1000000.0 (valeur de Qwen2.5)
        inv_freq = theta ** (-[0, 2] / 4) = [1.0, 0.001]

        moitiés (attendu)  : paires (canal 0, canal 2) et (canal 1, canal 3)
        entrelacé (faux)   : paires (canal 0, canal 1) et (canal 2, canal 3)

        q (1, num_heads=4,    seq=4, head_dim=4)  -> q_rope, même shape
        k (1, num_kv_heads=2, seq=4, head_dim=4)  -> k_rope, même shape
        position 0 : angle nul -> vecteur inchangé dans les deux conventions

    Ce que ce test vérifie
    ----------------------
    1. notre RoPE reproduit `apply_rotary_pos_emb` sur Q et sur K, shapes et dtype conservés,
       à rtol=1e-5 / atol=1e-6 ;
    2. la convention entrelacée, elle, NE reproduit PAS la référence : c'est bien le découpage
       par moitiés qui est imposé ;
    3. `theta` est lu dans la config (1000000.0 pour Qwen2.5) et non laissé à 10000.0 ;
    4. les propriétés géométriques restent vraies des deux côtés : identité en position 0 et
       norme de chaque vecteur de tête conservée.

    API à faire émerger (cible roadmap : « Qwen components », cible proposée : réutilisation de
    `src/inference_lab/nn/positional/rope.py` de la section 2.10)
    ------------------------------------------------------------
        def apply_rope(
            x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0
        ) -> torch.Tensor: ...

    Indice : côté référence, `Qwen2RotaryEmbedding(hf_config)(q, position_ids)` renvoie le
    couple `(cos, sin)` de shape (batch, seq, head_dim) — head_dim entier, car `cos` est la
    concaténation `cat((angles, angles), dim=-1)` ; puis
    `apply_rotary_pos_emb(q, k, cos, sin)` applique `q * cos + rotate_half(q) * sin`. Pièges :
    `position_ids` doit être de shape (batch, seq) côté Hugging Face alors que notre
    `apply_rope` prend un vecteur 1D de positions ; et la dimension de tête doit être en
    position 1 (batch, heads, seq, head_dim), sinon `cos` ne se diffuse pas sur le bon axe.
    """

    pytest.skip("Roadmap TDD 12.3 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.positional.rope import apply_rope

    # Arrange — référence Hugging Face locale, sans checkpoint (`from transformers import
    #           Qwen2Config` et `from transformers.models.qwen2.modeling_qwen2 import
    #           Qwen2RotaryEmbedding, apply_rotary_pos_emb`).
    #           `hf_config` : `Qwen2Config` réduite (hidden_size=16, num_hidden_layers=2,
    #           num_attention_heads=4, num_key_value_heads=2, intermediate_size=32, vocab_size=32)
    #           avec rope_theta=1000000.0, la valeur de Qwen2.5.
    #           `theta` : le flottant 1000000.0, passé explicitement à notre `apply_rope`.
    #           `q` : tenseur (1, 4, 4, 4) en `torch.float32`, déterministe
    #           (`torch.manual_seed(0)`), sans vecteur nul.
    #           `k` : tenseur (1, 2, 4, 4) en `torch.float32`, déterministe, différent de `q`.
    #           `positions` : tenseur d'entiers 1D valant 0, 1, 2, 3 (et sa version (1, 4) pour
    #           l'appel Hugging Face).
    #           `q_interleaved` : la MÊME rotation appliquée à `q` avec la convention entrelacée
    #           (paires de canaux voisins (0, 1) et (2, 3), mêmes angles), à construire à la main
    #           pour la comparer à la référence.

    # Act — côté référence, obtenir `cos, sin` avec `Qwen2RotaryEmbedding(hf_config)` puis
    #       `q_hf, k_hf = apply_rotary_pos_emb(q, k, cos, sin)` ; côté nous,
    #       `q_rope = apply_rope(q, positions, theta)` et
    #       `k_rope = apply_rope(k, positions, theta)`.

    # Assert 1 — égalité canal par canal avec la référence, sur Q et sur K
    assert q_rope.shape == (1, 4, 4, 4)
    assert k_rope.shape == (1, 2, 4, 4)
    assert q_rope.dtype is torch.float32
    torch.testing.assert_close(q_rope, q_hf, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(k_rope, k_hf, rtol=1e-5, atol=1e-6)

    # Assert 2 — la convention entrelacée a les bonnes shapes mais pas les bonnes valeurs
    assert q_interleaved.shape == q_hf.shape
    assert not torch.allclose(q_interleaved, q_hf, rtol=1e-3, atol=1e-3)

    # Assert 3 — theta est celui de la config : 1000000.0, pas la valeur par défaut 10000.0
    assert not torch.allclose(apply_rope(q, positions, 10000.0), q_hf, rtol=1e-3, atol=1e-3)

    # Assert 4 — propriétés géométriques communes : identité en 0, norme conservée
    torch.testing.assert_close(q_rope[:, :, 0], q[:, :, 0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(q_rope.norm(dim=-1), q.norm(dim=-1), rtol=1e-5, atol=1e-6)
