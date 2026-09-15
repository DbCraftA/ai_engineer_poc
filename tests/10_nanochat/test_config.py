"""Section 10.1 — configuration d'un modèle nanochat-like : déclarer TOUTE l'architecture.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/10_nanochat/test_config.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(32, 16, 4, 2, "rmsnorm", "rope", "swiglu") : c'est volontaire. Un test doit énoncer la
vérité attendue, pas la recalculer avec la même formule que le code testé.

Modèle jouet de la section 10, constant dans tout le chapitre : `vocab_size=32`,
`hidden_size=16`, `num_layers=2`, `num_heads=4`, `num_kv_heads=2`, `head_dim=4`,
`intermediate_size=32`, `seq=6`, `batch=1`. Il est volontairement plus gros que le jouet de
la section 3 (`vocab=16`, `hidden=8`, MHA à 2 têtes) : ici on ne construit plus un
transformer minimal mais une architecture MODERNE complète, celle de nanochat et de
Qwen2 — RMSNorm en pré-normalisation, RoPE, GQA (`num_kv_heads < num_heads`), MLP SwiGLU,
aucun biais, weight tying éventuel. La config de cette section est donc plus riche que
celle de 3.1 : elle déclare non seulement les dimensions, mais aussi les CHOIX
d'architecture, exactement comme un `config.json` Hugging Face.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import dataclasses

import pytest


@pytest.mark.tdd
@pytest.mark.model
def test_nanochat_like_config_defines_complete_architecture():
    """Roadmap 10.1 — une config moderne déclare les dimensions ET les choix d'architecture.

    Objectif d'apprentissage
    ------------------------
    Un `config.json` de LLM récent ne contient pas que des tailles : il fixe la famille
    d'architecture. Norme (`rmsnorm` plutôt que `layernorm`), placement de la norme
    (pré-normalisation plutôt que post), encodage positionnel (`rope` plutôt qu'un
    embedding de position appris), attention (GQA via `num_kv_heads`), activation du MLP
    (`swiglu` plutôt que GELU), présence de biais (aucun). Chacun de ces champs a une
    conséquence directe sur l'inférence : `num_kv_heads` divise la taille du KV cache,
    RoPE rend la position calculable au decode sans table, l'absence de biais retire un
    tenseur par projection. Déclarer ces choix dans la config, plutôt que les coder en
    dur dans les modules, est ce qui permettra plus tard de charger Qwen2.5-0.5B avec le
    MÊME code de modèle (section 12).

    La config est aussi le premier endroit où rejeter une architecture impossible : mieux
    vaut une `ValueError` à la construction qu'un `view()` qui explose au milieu du
    forward, ou pire, un `reshape` qui passe en mélangeant silencieusement les têtes.

    Schéma mental
    -------------
        NanochatLikeConfig(vocab_size=32, hidden_size=16, num_layers=2,
                           num_heads=4, num_kv_heads=2, intermediate_size=32)

            hidden_size=16 --- num_heads=4 ---> head_dim = 16 // 4 = 4
            16 = 4 x 4                          (division exacte OBLIGATOIRE)

            GQA : 4 têtes Q  ->  2 têtes KV     4 % 2 == 0, 2 requêtes par groupe KV
                  q0 q1 --> kv0        q2 q3 --> kv1

            choix déclarés : norm="rmsnorm" (pre), position="rope", mlp="swiglu",
                             biais=False    -> exactement la famille Llama / Qwen2

            hidden_size=16, num_heads=6  -> 16 / 6 non entier  -> ValueError
            num_heads=4, num_kv_heads=3  -> 4 % 3 != 0         -> ValueError

    Ce que ce test vérifie
    ----------------------
    1. la config est un dataclass qui déclare les six dimensions du modèle jouet ;
    2. elle déclare AUSSI les choix d'architecture : RMSNorm en pré-normalisation, RoPE,
       activation SwiGLU, et aucun biais dans l'attention ni dans le MLP ;
    3. les contraintes de cohérence tiennent : `head_dim = 4` dérivé, `hidden_size =
       num_heads * head_dim`, `num_heads % num_kv_heads == 0`, et GQA réellement active
       (`num_kv_heads < num_heads`, donc 2 requêtes par groupe KV) ;
    4. une config qui viole l'une de ces deux divisibilités est refusée à la construction
       par une `ValueError`.

    API à faire émerger (cible roadmap : `src/inference_lab/models/nanochat_like/config.py`)
    ---------------------------------------------------------------------------------------
        @dataclasses.dataclass
        class NanochatLikeConfig:
            vocab_size: int
            hidden_size: int
            num_layers: int
            num_heads: int
            num_kv_heads: int
            intermediate_size: int
            norm_type: str = "rmsnorm"
            norm_placement: str = "pre"
            position_encoding: str = "rope"
            activation: str = "swiglu"
            attention_bias: bool = False
            mlp_bias: bool = False
            tie_word_embeddings: bool = True
            rms_norm_eps: float = 1e-6
            rope_theta: float = 10000.0

            @property
            def head_dim(self) -> int: ...

    Indice : même schéma qu'en 3.1 — `__post_init__` pour les deux validations, `head_dim`
    en propriété dérivée et non en champ. Les valeurs par défaut des champs d'architecture
    portent le message pédagogique : le défaut d'un modèle 2024 n'est plus LayerNorm +
    positions apprises + biais, mais RMSNorm + RoPE + SwiGLU sans biais. Piège : ne
    confonds pas `num_heads` (têtes de requêtes, dimensionne Q et la sortie) et
    `num_kv_heads` (têtes de clés/valeurs, dimensionne K, V et le KV cache).
    """

    pytest.skip("Roadmap TDD 10.1 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.models.nanochat_like.config import NanochatLikeConfig

    # Arrange — construire `config`, une instance de `NanochatLikeConfig` pour le modèle
    #           jouet de la section 10 : vocab_size=32, hidden_size=16, num_layers=2,
    #           num_heads=4, num_kv_heads=2, intermediate_size=32 (soit head_dim=4, avec
    #           seq=6 et batch=1 pour les fichiers suivants). Laisser les champs
    #           d'architecture à leurs valeurs par défaut : c'est justement ce que
    #           l'assert 2 vérifie. Aucun tenseur ici : une config ne contient que des
    #           entiers, des booléens et des chaînes.

    # Act — rien à appeler : la construction de la config EST l'acte à tester ; les asserts
    #       lisent ensuite ses champs, sa dimension dérivée `head_dim` et son refus des
    #       combinaisons incohérentes.

    # Assert 1 — un dataclass qui déclare les dimensions du modèle
    assert dataclasses.is_dataclass(config)
    assert config.vocab_size == 32
    assert config.hidden_size == 16
    assert config.num_layers == 2
    assert config.num_heads == 4
    assert config.num_kv_heads == 2
    assert config.intermediate_size == 32

    # Assert 2 — les choix d'architecture sont déclarés, pas codés en dur dans les modules
    assert config.norm_type == "rmsnorm"
    assert config.norm_placement == "pre"
    assert config.position_encoding == "rope"
    assert config.activation == "swiglu"
    assert config.attention_bias is False
    assert config.mlp_bias is False

    # Assert 3 — contraintes de cohérence, et GQA réellement active
    assert config.head_dim == 4
    assert config.hidden_size == config.num_heads * config.head_dim
    assert config.hidden_size % config.num_heads == 0
    assert config.num_heads % config.num_kv_heads == 0
    assert config.num_kv_heads < config.num_heads
    assert config.num_heads // config.num_kv_heads == 2

    # Assert 4 — une architecture impossible est refusée dès la construction
    with pytest.raises(ValueError):
        NanochatLikeConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=2,
            num_heads=6,
            num_kv_heads=2,
            intermediate_size=32,
        )
    with pytest.raises(ValueError):
        NanochatLikeConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=2,
            num_heads=4,
            num_kv_heads=3,
            intermediate_size=32,
        )
