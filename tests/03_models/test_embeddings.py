"""Section 3.2 — embeddings : d'un id de token à un vecteur dense.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/03_models/test_embeddings.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
((16, 8), (1, 4, 8), ...) : c'est volontaire. Un test doit énoncer la vérité attendue,
pas la recalculer avec la même formule que le code testé.

Modèle jouet de la section 3, constant dans tout le fichier : `vocab_size=16`, `hidden=8`,
`num_layers=2`, `num_heads=2`, `seq=4`, `batch=1`. L'embedding est la toute première couche
du modèle : elle transforme des entiers (des indices, sans géométrie) en vecteurs sur
lesquels attention et MLP peuvent enfin calculer.

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
def test_embedding_maps_token_ids_to_hidden_vectors():
    """Roadmap 3.2 — une table d'embedding est une lecture de lignes, pas un calcul.

    Objectif d'apprentissage
    ------------------------
    Le tokenizer produit des entiers ; le réseau a besoin de vecteurs. La table
    d'embedding est une matrice `(vocab_size, hidden)` dont la ligne `i` EST la
    représentation du token `i`. Deux conséquences pour un moteur d'inférence :
    l'opération est un `gather` (limité par la bande passante mémoire, pas par le calcul),
    et la table est souvent l'un des plus gros tenseurs de poids du modèle — sur
    Qwen2.5-0.5B, `151936 x 896` paramètres, soit un quart du modèle. C'est aussi pour
    cela que beaucoup de modèles partagent ce tenseur avec le `lm_head` (weight tying,
    section 3.5).

    Schéma mental
    -------------
        table (vocab_size=16, hidden=8)          token_ids (batch=1, seq=4)
        ligne 0  [ . . . . . . . . ]             [[ a, b, a, c ]]
        ligne 1  [ . . . . . . . . ]                  |
        ...                                           v  lecture des lignes a, b, a, c
        ligne 15 [ . . . . . . . . ]             hidden_states (1, 4, 8)

        les positions 0 et 2 portent le MÊME id `a` -> MÊME vecteur de 8 valeurs
        (aucune notion de position ici : c'est le rôle de RoPE, section 2.10)

    Ce que ce test vérifie
    ----------------------
    1. la table de poids a exactement la shape `(vocab_size, hidden) = (16, 8)` ;
    2. l'indexation par `token_ids (1, 4)` produit des états cachés `(1, 4, 8)` en float32 ;
    3. chaque position reçoit exactement la ligne de la table correspondant à son id, donc
       deux positions portant le même id reçoivent le même vecteur ;
    4. l'embedding est équivalent à `one_hot(token_ids) @ table` : c'est bien une projection
       linéaire dégénérée, implémentée comme une lecture d'index.

    API à faire émerger (cible roadmap : `src/inference_lab/nn/embeddings.py`)
    ------------------------------------------------------------------------
        class TokenEmbedding(torch.nn.Module):
            def __init__(self, vocab_size: int, hidden_size: int) -> None: ...
            weight: torch.Tensor  # (vocab_size, hidden_size)
            def forward(self, token_ids: torch.Tensor) -> torch.Tensor: ...

    Indice : `torch.nn.Embedding` fait déjà exactement cela, et son attribut `weight` est
    la table. Écris-le d'abord à la main (`self.weight[token_ids]` sur un
    `torch.nn.Parameter`) pour voir que l'indexation avancée d'un tenseur `(B, S)` sur une
    table `(V, H)` rend directement `(B, S, H)`. Piège : `token_ids` doit être un tenseur
    entier (`torch.long`), pas un float.
    """

    pytest.skip("Roadmap TDD 3.2 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.embeddings import TokenEmbedding

    # Arrange — fixer la graine (`torch.manual_seed(0)`) puis construire :
    #           - `embedding`, un `TokenEmbedding` de vocab_size=16 et hidden_size=8 ;
    #           - `token_ids`, un tenseur `torch.long` de shape (batch=1, seq=4) dont tous les
    #             ids sont dans [0, 16) et dont les positions 0 et 2 portent le MÊME id,
    #             les deux autres positions portant des ids différents de celui-là.

    # Act — passer `token_ids` dans `embedding` pour obtenir `hidden_states`.

    # Assert 1 — la table est bien une matrice (vocab_size, hidden)
    assert isinstance(embedding, TokenEmbedding)
    assert embedding.weight.shape == (16, 8)

    # Assert 2 — l'indexation ajoute la dimension cachée sans toucher batch ni seq
    assert hidden_states.shape == (1, 4, 8)
    assert hidden_states.dtype is torch.float32

    # Assert 3 — chaque position reçoit la ligne de son id ; ids identiques -> vecteurs identiques
    torch.testing.assert_close(hidden_states[0, 0], embedding.weight[token_ids[0, 0]])
    torch.testing.assert_close(hidden_states[0, 0], hidden_states[0, 2])
    assert not torch.equal(hidden_states[0, 0], hidden_states[0, 1])

    # Assert 4 — équivalence avec la formulation matricielle one-hot (l'oracle du concept)
    one_hot = torch.nn.functional.one_hot(token_ids, num_classes=16).to(torch.float32)
    torch.testing.assert_close(hidden_states, one_hot @ embedding.weight, rtol=1e-5, atol=1e-6)
