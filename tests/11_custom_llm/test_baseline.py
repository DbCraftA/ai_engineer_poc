"""Section 11.1 — le modèle configurable : ce qui est déclaré est ce qui est construit.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/11_custom_llm/test_baseline.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(2 blocs, logits (1, 6, 32), "mha", "rmsnorm", "naive") : c'est volontaire. Un test doit
énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Dimensions jouets constantes de toute la section 11 : vocab=32, hidden=16, num_layers=2,
num_heads=4, num_kv_heads=4 en MHA (2 en GQA), head_dim=4, intermediate=32, seq=6, batch=1.
La section 11 est le laboratoire d'expérimentation du dépôt : le modèle y est CONFIGURABLE
pour rendre les choix d'architecture interchangeables et donc COMPARABLES (MHA vs GQA,
RMSNorm vs LayerNorm, attention naïve vs SDPA). Elle réutilise les briques des sections 2
(attention, normalisation) et 8 (SDPA) ; ce premier test pose l'invariant qui rend toutes
les comparaisons suivantes crédibles : le modèle construit correspond EXACTEMENT à la
configuration déclarée.

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
def test_custom_llm_baseline_matches_declared_configuration():
    """Roadmap 11.1 — la configuration est la source de vérité du modèle instancié.

    Objectif d'apprentissage
    ------------------------
    Un laboratoire d'architecture ne vaut rien si l'objet construit ne correspond pas à ce
    qu'on a demandé. Comparer « GQA contre MHA » n'a de sens que si l'on peut PROUVER que le
    modèle testé utilise bien GQA. D'où la règle de cette section : chaque choix
    d'architecture est déclaré une seule fois dans un objet de configuration, et reste
    LISIBLE sur le modèle instancié (`model.config`). C'est exactement le rôle des
    `PretrainedConfig` de Hugging Face, et c'est ce qui permettra plus tard de charger
    Qwen2.5-0.5B en ne changeant que des nombres : 24 couches, 14 têtes de query, 2 têtes KV,
    hidden 896. Ici, les dimensions sont jouets pour que le test tourne en quelques
    millisecondes sur CPU, mais le contrat est le même.

    Schéma mental
    -------------
        config : vocab=32, hidden=16, num_layers=2, num_heads=4, num_kv_heads=4,
                 head_dim=4, intermediate=32, norm="rmsnorm", backend="naive"

        input_ids (batch=1, seq=6)  --embed (32, 16)-->  x (1, 6, 16)
              -> bloc 0 -> bloc 1  (num_layers=2)       -> x (1, 6, 16)
              --final_norm--> --lm_head (16, 32)-->     logits (1, 6, 32)

        num_heads=4 x head_dim=4 = 16 = hidden : les projections sont carrées (16, 16)
        num_kv_heads == num_heads == 4  ->  attention_type == "mha"

    Ce que ce test vérifie
    ----------------------
    1. le nombre de blocs construits vaut exactement `num_layers` = 2, et la configuration
       portée par le modèle est celle qu'on a déclarée ;
    2. les quatre choix d'architecture sont lisibles sur le modèle : type d'attention
       ("mha"), type de normalisation ("rmsnorm"), backend d'attention ("naive"), et le
       module de normalisation réellement créé est bien un `RMSNorm` ;
    3. la passe avant produit des logits de shape (1, 6, 32) = (batch, seq, vocab), en
       float32 et sans valeur non finie ;
    4. les dimensions internes sont cohérentes : `num_heads * head_dim == hidden_size`, donc
       la projection Q pèse 16 x 16 = 256 paramètres.

    API à faire émerger (cible roadmap : `src/inference_lab/models/custom_llm/`, cibles
    proposées `src/inference_lab/models/custom_llm/config.py` et
    `src/inference_lab/models/custom_llm/model.py`)
    -------------------------------------------------------------------------------------
        @dataclass
        class CustomLLMConfig:
            vocab_size: int
            hidden_size: int
            num_layers: int
            num_heads: int
            num_kv_heads: int
            head_dim: int
            intermediate_size: int
            norm_type: str = "rmsnorm"
            attention_backend: str = "naive"
            @property
            def attention_type(self) -> str: ...   # "mha" si num_kv_heads == num_heads

        class CustomLLM(torch.nn.Module):
            def __init__(self, config: CustomLLMConfig) -> None: ...
            def forward(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    Indice : garde le modèle minimal — `torch.nn.Embedding`, une `torch.nn.ModuleList` de
    blocs nommée `blocks`, une normalisation finale, un `lm_head` linéaire sans biais. Stocke
    la configuration reçue dans `self.config` sans la recopier champ par champ : c'est ce qui
    rend l'assert 1 vrai par construction. Piège : construire les blocs avec une boucle
    `range(2)` codée en dur au lieu de `range(config.num_layers)` — le test 11.1 ne le verrait
    pas, mais toute la section 11 reposerait sur une configuration décorative.
    """

    pytest.skip("Roadmap TDD 11.1 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.models.custom_llm.config import CustomLLMConfig
    from inference_lab.models.custom_llm.model import CustomLLM

    # Arrange — dimensions jouets de la section : vocab=32, hidden=16, num_layers=2,
    #           num_heads=4, num_kv_heads=4 (donc MHA), head_dim=4, intermediate=32,
    #           seq=6, batch=1.
    #           `config` : une `CustomLLMConfig` déclarant EXACTEMENT ces dimensions, avec
    #           `norm_type="rmsnorm"` et `attention_backend="naive"`.
    #           `model` : le `CustomLLM` construit à partir de `config`, en `torch.float32`,
    #           sur CPU, puis passé en `eval()`.
    #           `input_ids` : tenseur d'entiers (1, 6) dont toutes les valeurs sont dans
    #           [0, 32), déterministe (seed fixée).

    # Act — appeler `logits = model(input_ids)` pour la passe avant complète.

    # Assert 1 — autant de blocs que déclaré, et la configuration est celle qu'on a donnée
    assert isinstance(model, CustomLLM)
    assert isinstance(config, CustomLLMConfig)
    assert len(model.blocks) == 2
    assert model.config is config
    assert model.config.num_layers == 2
    assert model.config.hidden_size == 16
    assert model.config.vocab_size == 32

    # Assert 2 — les choix d'architecture sont lisibles sur le modèle instancié
    assert model.config.attention_type == "mha"
    assert model.config.num_kv_heads == model.config.num_heads == 4
    assert model.config.norm_type == "rmsnorm"
    assert model.config.attention_backend == "naive"
    assert type(model.blocks[0].input_norm).__name__ == "RMSNorm"

    # Assert 3 — la sortie est un jeu de logits (batch, seq, vocab)
    assert logits.shape == (1, 6, 32)
    assert logits.dtype is torch.float32
    assert bool(logits.isfinite().all())

    # Assert 4 — cohérence dimensionnelle : num_heads x head_dim == hidden_size
    assert model.config.num_heads * model.config.head_dim == 16
    assert model.blocks[0].attention.q_proj.weight.numel() == 256
