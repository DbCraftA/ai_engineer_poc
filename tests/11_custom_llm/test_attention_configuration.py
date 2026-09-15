"""Section 11.2 — MHA ou GQA au choix : ce que le nombre de têtes KV coûte vraiment.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/11_custom_llm/test_attention_configuration.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(k_proj : 256 paramètres en MHA contre 128 en GQA, KV cache 256 octets par token contre
128) : c'est volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec
la même formule que le code testé.

Dimensions jouets constantes de toute la section 11 : vocab=32, hidden=16, num_layers=2,
num_heads=4, num_kv_heads=4 en MHA (2 en GQA), head_dim=4, intermediate=32, seq=6, batch=1.
Ce test transforme un choix de configuration en conséquence MESURABLE : basculer MHA -> GQA
ne change ni les shapes de sortie ni la mathématique de l'attention (section 2.8), mais
divise par 2 la taille des projections K et V et la taille du KV cache par token. C'est le
premier arbitrage que le laboratoire de la section 11 rend chiffrable.

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
def test_custom_llm_can_switch_between_mha_and_gqa():
    """Roadmap 11.2 — un seul champ de configuration, deux architectures d'attention.

    Objectif d'apprentissage
    ------------------------
    MHA et GQA ne diffèrent que par un nombre : `num_kv_heads`. Quand il vaut `num_heads`,
    chaque tête de query a sa propre paire (K, V) : c'est la MHA. Quand il est plus petit,
    plusieurs têtes de query partagent la même paire : c'est la GQA (section 2.8). Le point
    pédagogique de ce test n'est pas la mathématique, déjà couverte, mais le PRIX : les
    matrices de projection K et V rétrécissent exactement dans le ratio
    `num_kv_heads / num_heads`, et le KV cache par token avec elles. En decode, ce cache est
    relu intégralement à chaque token : sa taille fixe le débit. Qwen2.5-0.5B applique le même
    calcul avec 14 têtes de query pour 2 têtes KV, soit 7 fois moins de cache — c'est ce
    facteur, pas une astuce de kernel, qui rend les longs contextes tenables.

    Schéma mental
    -------------
        hidden=16, num_heads=4, head_dim=4, num_layers=2, fp32 (4 octets)

        MHA (num_kv_heads=4) : k_proj (16, 16) -> 16 x 4 x 16 = 256 paramètres
        GQA (num_kv_heads=2) : k_proj ( 8, 16) -> 16 x 2 x  4 = 128 paramètres   (÷2)

        q_proj et o_proj restent (16, 16) = 256 dans les DEUX cas : seuls K et V bougent.

        KV cache par token = 2 (K et V) x num_layers x num_kv_heads x head_dim x 4 octets
            MHA : 2 x 2 x 4 x 4 x 4 = 256 octets/token
            GQA : 2 x 2 x 2 x 4 x 4 = 128 octets/token                            (÷2)

        Économie totale de paramètres = 2 couches x (K et V) x 128 = 512 paramètres.

    Ce que ce test vérifie
    ----------------------
    1. le type d'attention se lit sur le modèle : "mha" quand num_kv_heads == num_heads,
       "gqa" quand num_kv_heads < num_heads ;
    2. la conséquence chiffrée sur les poids : `k_proj` passe de hidden x hidden = 256 à
       hidden x num_kv_heads x head_dim = 128 paramètres, tandis que `q_proj` et `o_proj`
       restent à 256 — soit 512 paramètres de moins au total sur les 2 couches ;
    3. le KV cache par token diminue dans le MÊME ratio : 256 octets contre 128 en fp32 ;
    4. la bascule est transparente vue de l'extérieur : les deux modèles acceptent les mêmes
       entrées et rendent des logits (1, 6, 32) finis.

    API à faire émerger (cible roadmap : « custom model », cibles proposées
    `src/inference_lab/models/custom_llm/config.py` et
    `src/inference_lab/models/custom_llm/model.py`)
    ---------------------------------------------------------------------
        class CustomLLMConfig:
            num_heads: int
            num_kv_heads: int
            @property
            def attention_type(self) -> str: ...           # "mha" ou "gqa"
            def kv_cache_bytes_per_token(self, dtype: torch.dtype = torch.float32) -> int: ...

    Indice : ne crée pas deux classes d'attention. Une seule suffit : dimensionne `k_proj` et
    `v_proj` en `torch.nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)` et aligne
    les têtes au moment du calcul avec `repeat_kv_heads` de la section 2.8. `q_proj` et
    `o_proj` gardent `num_heads * head_dim`. Pièges : dimensionner K et V sur `num_heads` puis
    « jeter » des têtes (l'économie disparaît), et matérialiser les têtes répétées dans le
    cache (le gain mémoire disparaît aussi). Refuse une configuration où `num_heads` n'est pas
    divisible par `num_kv_heads`.
    """

    pytest.skip("Roadmap TDD 11.2 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.models.custom_llm.config import CustomLLMConfig
    from inference_lab.models.custom_llm.model import CustomLLM

    # Arrange — dimensions jouets de la section : vocab=32, hidden=16, num_layers=2,
    #           num_heads=4, head_dim=4, intermediate=32, seq=6, batch=1.
    #           `config_mha` : une `CustomLLMConfig` avec `num_kv_heads=4` (== num_heads).
    #           `config_gqa` : la MÊME configuration, seul `num_kv_heads=2` change.
    #           `mha` et `gqa` : les deux `CustomLLM` construits depuis ces configurations,
    #           en `torch.float32`, sur CPU, en `eval()`.
    #           `input_ids` : tenseur d'entiers (1, 6) à valeurs dans [0, 32), déterministe
    #           (seed fixée), partagé par les deux modèles.

    # Act — récupérer les poids des projections du premier bloc de chaque modèle
    #       (`k_proj`, `q_proj`, `o_proj`), compter le total de paramètres de chaque modèle
    #       dans `params_mha` et `params_gqa`, interroger `kv_cache_bytes_per_token()` sur les
    #       deux configurations, puis calculer `logits_mha` et `logits_gqa` sur `input_ids`.

    # Assert 1 — le choix d'architecture est lisible, et c'est num_kv_heads qui le décide
    assert isinstance(mha, CustomLLM)
    assert isinstance(config_gqa, CustomLLMConfig)
    assert mha.config.attention_type == "mha"
    assert gqa.config.attention_type == "gqa"
    assert mha.config.num_kv_heads == mha.config.num_heads == 4
    assert gqa.config.num_kv_heads == 2
    assert gqa.config.num_kv_heads < gqa.config.num_heads

    # Assert 2 — les projections K et V rétrécissent dans le ratio num_kv_heads / num_heads
    assert mha.blocks[0].attention.k_proj.weight.numel() == 256
    assert gqa.blocks[0].attention.k_proj.weight.numel() == 128
    assert gqa.blocks[0].attention.v_proj.weight.numel() == 128
    assert gqa.blocks[0].attention.k_proj.weight.shape == (8, 16)
    assert gqa.blocks[0].attention.q_proj.weight.numel() == 256
    assert gqa.blocks[0].attention.o_proj.weight.numel() == 256
    assert params_mha - params_gqa == 512

    # Assert 3 — le KV cache par token suit exactement le même ratio
    assert mha.config.kv_cache_bytes_per_token(torch.float32) == 256
    assert gqa.config.kv_cache_bytes_per_token(torch.float32) == 128
    assert gqa.config.kv_cache_bytes_per_token(torch.float32) * 2 == 256

    # Assert 4 — bascule transparente : même contrat d'entrée et de sortie
    assert logits_mha.shape == (1, 6, 32)
    assert logits_gqa.shape == (1, 6, 32)
    assert bool(logits_gqa.isfinite().all())
