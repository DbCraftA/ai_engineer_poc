"""Section 3.1 — configuration d'un transformer minimal (les dimensions comme donnée).

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/03_models/test_minimal_config.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(16, 8, 2, 4, ...) : c'est volontaire. Un test doit énoncer la vérité attendue,
pas la recalculer avec la même formule que le code testé.

Toute la section 3 travaille sur le MÊME modèle jouet, minuscule et constant d'un fichier
à l'autre : `vocab_size=16`, `hidden=8`, `num_layers=2`, `num_heads=2`, `seq=4`, `batch=1`.
La config posée ici est la source unique de ces dimensions : embeddings, blocs, lm_head et
KV cache s'y référeront tous, exactement comme `Qwen2Config` pilote Qwen2.5-0.5B.

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
def test_minimal_transformer_config_defines_model_dimensions():
    """Roadmap 3.1 — une config est un contrat de dimensions, pas un sac de paramètres.

    Objectif d'apprentissage
    ------------------------
    Dans un moteur d'inférence, aucune shape n'est écrite en dur : tout dérive de la
    config du modèle. `hidden_size`, `num_layers` et `num_heads` déterminent la taille
    des poids, la taille du KV cache par token et le découpage en têtes de l'attention.
    Poser ce dataclass maintenant évite de recalculer `head_dim` à seize endroits plus
    tard, et permet de rejeter TÔT une config incohérente : si `hidden_size` n'est pas
    divisible par `num_heads`, le `view(batch, seq, num_heads, head_dim)` de la MHA
    (section 2.7) est impossible. Mieux vaut échouer à la construction de la config
    qu'au milieu d'un forward.

    Schéma mental
    -------------
        MinimalTransformerConfig(vocab_size=16, hidden_size=8, num_layers=2, num_heads=2)

            hidden_size=8 ---- decoupe en num_heads=2 ----> head_dim = 8 // 2 = 4
            8 = 2 x 4                                        (division exacte OBLIGATOIRE)

            hidden_size=8, num_heads=3  ->  8 / 3 non entier  ->  ValueError

    Ce que ce test vérifie
    ----------------------
    1. la config est bien un dataclass et expose les quatre dimensions demandées ;
    2. `head_dim` est la dimension par tête, cohérente avec `hidden_size = num_heads * head_dim` ;
    3. une config dont `hidden_size` n'est pas divisible par `num_heads` est refusée
       à la construction, avec une `ValueError`.

    API à faire émerger (cible roadmap : `src/inference_lab/models/minimal_transformer/config.py`)
    ---------------------------------------------------------------------------------------------
        @dataclasses.dataclass
        class MinimalTransformerConfig:
            vocab_size: int
            hidden_size: int
            num_layers: int
            num_heads: int

            @property
            def head_dim(self) -> int: ...

    Indice : `dataclasses.dataclass` génère `__init__` pour toi ; la validation se place dans
    `__post_init__` (c'est le seul endroit appelé après l'affectation des champs). `head_dim`
    n'est PAS un champ : c'est une propriété dérivée, sinon deux sources de vérité peuvent
    se contredire.
    """

    pytest.skip("Roadmap TDD 3.1 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.models.minimal_transformer.config import MinimalTransformerConfig

    # Arrange — construire `config`, une instance de `MinimalTransformerConfig` pour le modèle
    #           jouet de la section : vocab_size=16, hidden_size=8, num_layers=2, num_heads=2.
    #           Aucun tenseur ici : une config ne contient que des entiers.

    # Act — rien à appeler : la construction de la config EST l'acte à tester ; les asserts
    #       lisent ensuite ses attributs et sa dimension dérivée `head_dim`.

    # Assert 1 — c'est un dataclass, et il déclare les dimensions du modèle
    assert dataclasses.is_dataclass(config)
    assert config.vocab_size == 16
    assert config.hidden_size == 8
    assert config.num_layers == 2
    assert config.num_heads == 2

    # Assert 2 — la dimension par tête est dérivée, pas déclarée
    assert config.head_dim == 4
    assert config.hidden_size == config.num_heads * config.head_dim

    # Assert 3 — contrainte de cohérence : hidden_size doit être divisible par num_heads
    assert config.hidden_size % config.num_heads == 0
    with pytest.raises(ValueError):
        MinimalTransformerConfig(vocab_size=16, hidden_size=8, num_layers=2, num_heads=3)
