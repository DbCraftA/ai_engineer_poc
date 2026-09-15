"""Section 11.3 — choisir l'implémentation de normalisation par la configuration.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/11_custom_llm/test_normalization_configuration.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
("rmsnorm" / "layernorm", présence ou absence d'un paramètre `bias`, logits (1, 6, 32),
`ValueError` sur une valeur inconnue) : c'est volontaire. Un test doit énoncer la vérité
attendue, pas la recalculer avec la même formule que le code testé.

Dimensions jouets constantes de toute la section 11 : vocab=32, hidden=16, num_layers=2,
num_heads=4, num_kv_heads=4 en MHA (2 en GQA), head_dim=4, intermediate=32, seq=6, batch=1.
Contrairement au backend d'attention (11.4 / 11.5), la normalisation n'est PAS un choix
neutre : RMSNorm et LayerNorm calculent deux choses différentes (section 2.11). Le test doit
donc prouver deux propriétés à la fois : la fabrique choisit le bon module, et les sorties
diffèrent réellement. Une valeur inconnue doit échouer tout de suite, pas retomber
silencieusement sur un défaut.

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
def test_custom_llm_can_select_normalization_implementation():
    """Roadmap 11.3 — une chaîne de configuration décide de la classe réellement instanciée.

    Objectif d'apprentissage
    ------------------------
    C'est le patron « fabrique » (factory) : la configuration porte un nom, le modèle construit
    l'objet correspondant. Il donne au laboratoire sa valeur — on compare deux normalisations
    en changeant une chaîne, sans dupliquer le modèle. Il impose aussi une discipline :
    une valeur inconnue doit lever une erreur immédiatement. Un défaut silencieux
    (`else: return RMSNorm(...)`) produirait le pire scénario possible : une faute de frappe
    dans `norm_type` donnerait un modèle qui tourne, qui converge, et dont les résultats de
    comparaison sont faux. C'est exactement ce mécanisme de dispatch qui permettra plus tard de
    charger Qwen (RMSNorm) et un modèle de type GPT-2 (LayerNorm) avec le même code de modèle,
    et de mesurer ce que coûte le centrage de LayerNorm.

    Schéma mental
    -------------
        config.norm_type = "rmsnorm"   -> RMSNorm(16)            : poids (16,), pas de biais
                                          y = x / sqrt(mean(x^2) + eps) * weight
        config.norm_type = "layernorm" -> torch.nn.LayerNorm(16) : poids (16,) + biais (16,)
                                          y = (x - mean) / sqrt(var + eps) * weight + bias
        config.norm_type = "batchnorm"  -> ValueError

        même poids d'attention et de MLP, deux normalisations :
            logits (1, 6, 32) des deux côtés, mais valeurs DIFFÉRENTES
            (LayerNorm soustrait la moyenne, RMSNorm non)

    Ce que ce test vérifie
    ----------------------
    1. le nom déclaré se lit sur le modèle et le module instancié est de la classe voulue :
       `RMSNorm` d'un côté, `torch.nn.LayerNorm` de l'autre, à chacun des 2 blocs ;
    2. la différence est structurelle et pas seulement nominale : LayerNorm apporte un
       paramètre de biais par normalisation, RMSNorm n'en a aucun ;
    3. les deux modèles produisent des logits (1, 6, 32) finis, mais NUMÉRIQUEMENT
       différents : le choix de normalisation change le calcul ;
    4. une valeur inconnue de `norm_type` lève une `ValueError` : aucun repli silencieux.

    API à faire émerger (cible roadmap : « custom model », cibles proposées
    `src/inference_lab/models/custom_llm/config.py` et
    `src/inference_lab/models/custom_llm/model.py`)
    ---------------------------------------------------------------------
        class CustomLLMConfig:
            norm_type: str = "rmsnorm"      # "rmsnorm" | "layernorm", validé à la création

        class RMSNorm(torch.nn.Module):     # enveloppe module du `rms_norm` de la section 2.11
            def __init__(self, hidden_size: int, eps: float = 1e-6) -> None: ...

        def build_norm(config: CustomLLMConfig) -> torch.nn.Module: ...

    Indice : valide `norm_type` dès la construction de la configuration (`__post_init__` d'une
    `dataclass`) — c'est là que l'erreur est la plus utile, avant qu'un seul poids ne soit
    alloué. `build_norm` se réduit alors à un dictionnaire `{"rmsnorm": ..., "layernorm": ...}`.
    Réutilise `rms_norm` de `nn/normalization/rmsnorm.py` dans `RMSNorm.forward` plutôt que de
    réécrire la formule. Pièges : renvoyer une seule instance partagée par tous les blocs (les
    poids appris seraient liés entre couches — il faut un module NEUF par emplacement), et
    lever `KeyError` ou `AssertionError` au lieu de `ValueError`.
    """

    pytest.skip("Roadmap TDD 11.3 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.models.custom_llm.config import CustomLLMConfig
    from inference_lab.models.custom_llm.model import CustomLLM

    # Arrange — dimensions jouets de la section : vocab=32, hidden=16, num_layers=2,
    #           num_heads=4, num_kv_heads=4, head_dim=4, intermediate=32, seq=6, batch=1.
    #           `model_rms` : un `CustomLLM` construit avec `norm_type="rmsnorm"`.
    #           `model_ln` : le MÊME jeu de dimensions avec `norm_type="layernorm"`, les autres
    #           poids initialisés depuis la même seed pour que la seule différence observable
    #           soit la normalisation.
    #           Les deux modèles sont en `torch.float32`, sur CPU, en `eval()`.
    #           `input_ids` : tenseur d'entiers (1, 6) à valeurs dans [0, 32), déterministe
    #           (seed fixée), partagé par les deux modèles.
    #           `bias_names_rms`, `bias_names_ln` : les noms de paramètres de chaque modèle qui
    #           se terminent par `input_norm.bias`, obtenus via `named_parameters()`.

    # Act — calculer `logits_rms = model_rms(input_ids)` et `logits_ln = model_ln(input_ids)`,
    #       et récupérer les modules de normalisation d'entrée des deux blocs de chaque modèle.

    # Assert 1 — le nom déclaré et la classe instanciée concordent, dans les 2 blocs
    assert isinstance(model_rms, CustomLLM)
    assert model_rms.config.norm_type == "rmsnorm"
    assert model_ln.config.norm_type == "layernorm"
    assert type(model_rms.blocks[0].input_norm).__name__ == "RMSNorm"
    assert type(model_rms.blocks[1].input_norm).__name__ == "RMSNorm"
    assert isinstance(model_ln.blocks[0].input_norm, torch.nn.LayerNorm)
    assert not isinstance(model_rms.blocks[0].input_norm, torch.nn.LayerNorm)

    # Assert 2 — différence structurelle : LayerNorm a un biais, RMSNorm n'en a pas
    assert bias_names_rms == []
    assert len(bias_names_ln) == 2
    assert model_rms.blocks[0].input_norm.weight.shape == (16,)
    assert model_ln.blocks[0].input_norm.bias.shape == (16,)

    # Assert 3 — même contrat de sortie, mais deux calculs différents
    assert logits_rms.shape == (1, 6, 32)
    assert logits_ln.shape == (1, 6, 32)
    assert bool(logits_rms.isfinite().all())
    assert not torch.allclose(logits_rms, logits_ln, rtol=1e-3, atol=1e-3)

    # Assert 4 — valeur inconnue : échec immédiat, pas de repli silencieux
    with pytest.raises(ValueError):
        CustomLLMConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=4,
            intermediate_size=32,
            norm_type="batchnorm",
        )
