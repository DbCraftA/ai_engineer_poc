"""Section 12.1 — traduire la configuration Hugging Face de Qwen2 en configuration interne.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/12_qwen/test_config_mapping.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(hidden=16, 4 têtes de query, 2 têtes KV, head_dim=4) : c'est volontaire. Un test doit énoncer
la vérité attendue, pas la recalculer avec la même formule que le code testé.

Toute la section 12 compare NOTRE implémentation à la référence Hugging Face
`Qwen/Qwen2.5-0.5B-Instruct`. Ce premier test n'a besoin d'AUCUN poids : une
`transformers.Qwen2Config` réduite suffit, elle porte exactement les mêmes noms de champs que
le vrai checkpoint. Sans cette traduction champ par champ, tous les tests suivants (RMSNorm,
RoPE, GQA, MLP, poids, logits) comparent deux architectures différentes et échouent sans
raison compréhensible.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest


@pytest.mark.tdd
@pytest.mark.model
@pytest.mark.hf
def test_qwen_config_maps_to_internal_model_dimensions():
    """Roadmap 12.1 — les noms de champs de Hugging Face deviennent notre configuration.

    Objectif d'apprentissage
    ------------------------
    Un moteur d'inférence ne commence pas par du calcul, il commence par une configuration :
    c'est elle qui décide de la taille des poids à charger, du nombre de couches à empiler, de
    la taille du KV cache par token et du découpage en têtes. Hugging Face fixe le vocabulaire
    de noms (`hidden_size`, `num_key_value_heads`, `rope_theta`, ...) ; notre code doit le
    traduire une fois pour toutes, sans le réinterpréter. Une seule erreur de traduction ici
    (par exemple confondre `num_attention_heads` et `num_key_value_heads`) se propage en
    tenseurs de shapes valides mais de contenus faux, ce qui est le pire des bugs.

    Schéma mental
    -------------
        config réduite du test (locale, aucun téléchargement) :
            hidden_size=16, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=32, vocab_size=32
            head_dim = hidden_size // num_attention_heads = 16 // 4 = 4
            groupe GQA = 4 // 2 = 2 têtes de query par tête KV

        vrai Qwen2.5-0.5B-Instruct, mêmes champs, autres nombres :
            hidden_size=896, num_hidden_layers=24, num_attention_heads=14,
            num_key_value_heads=2, intermediate_size=4864, vocab_size=151936
            head_dim = 896 // 14 = 64, groupe GQA = 7

    Ce que ce test vérifie
    ----------------------
    1. la correspondance champ par champ entre la config Hugging Face et la nôtre ;
    2. `head_dim` est DÉDUIT (hidden_size // num_attention_heads = 4), pas recopié ;
    3. les champs numériques non dimensionnels sont transmis eux aussi : `rms_norm_eps`,
       `rope_theta` et `tie_word_embeddings` ;
    4. une configuration GQA incohérente (têtes de query non divisibles par têtes KV) est
       refusée explicitement au lieu de casser plus tard dans l'attention.

    API à faire émerger (cible roadmap : `src/inference_lab/models/qwen/config.py`)
    -----------------------------------------------------------------------------
        @dataclass(frozen=True)
        class QwenConfig:
            hidden_size: int
            num_hidden_layers: int
            num_attention_heads: int
            num_key_value_heads: int
            intermediate_size: int
            rms_norm_eps: float
            rope_theta: float
            vocab_size: int
            tie_word_embeddings: bool

            @property
            def head_dim(self) -> int: ...

        def from_hf_config(hf_config) -> QwenConfig: ...

    Indice : `transformers.Qwen2Config` accepte tous ces champs en mots-clés, donc la
    référence est locale, exacte et instantanée. Attention, les valeurs par défaut de
    `Qwen2Config` ne sont PAS celles du checkpoint 0.5B (`rope_theta` vaut 10000.0 par défaut
    contre 1000000.0 chez Qwen2.5, `tie_word_embeddings` vaut False contre True) : passe-les
    explicitement. Autre piège : `Qwen2Config` n'expose pas d'attribut `head_dim`, c'est à
    `QwenConfig` de le calculer, et `hidden_size % num_attention_heads` doit valoir 0.
    """

    pytest.skip("Roadmap TDD 12.1 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.models.qwen.config import QwenConfig, from_hf_config

    # Arrange — référence Hugging Face locale, sans poids ni téléchargement
    #           (`from transformers import Qwen2Config`).
    #           `hf_config` : une `Qwen2Config` construite avec hidden_size=16,
    #           num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    #           intermediate_size=32, vocab_size=32, rms_norm_eps=1e-6, rope_theta=1000000.0,
    #           tie_word_embeddings=True (les deux derniers sont les valeurs de Qwen2.5).
    #           `hf_config_bad` : la même config mais avec num_key_value_heads=3, donc 4 têtes
    #           de query non divisibles par 3 têtes KV.

    # Act — traduire `hf_config` en configuration interne : `config = from_hf_config(hf_config)`.

    # Assert 1 — correspondance champ par champ, sans réinterprétation
    assert isinstance(config, QwenConfig)
    assert config.hidden_size == hf_config.hidden_size == 16
    assert config.num_hidden_layers == hf_config.num_hidden_layers == 2
    assert config.num_attention_heads == hf_config.num_attention_heads == 4
    assert config.num_key_value_heads == hf_config.num_key_value_heads == 2
    assert config.intermediate_size == hf_config.intermediate_size == 32
    assert config.vocab_size == hf_config.vocab_size == 32

    # Assert 2 — head_dim est déduit de hidden_size et du nombre de têtes de query
    assert config.head_dim == 4
    assert config.head_dim == config.hidden_size // config.num_attention_heads
    assert config.head_dim * config.num_attention_heads == 16

    # Assert 3 — les champs non dimensionnels comptent autant que les dimensions
    assert config.rms_norm_eps == 1e-6
    assert config.rope_theta == 1000000.0
    assert config.tie_word_embeddings is True

    # Assert 4 — 4 têtes de query pour 3 têtes KV : configuration GQA impossible
    assert config.num_attention_heads % config.num_key_value_heads == 0
    with pytest.raises(ValueError):
        from_hf_config(hf_config_bad)
