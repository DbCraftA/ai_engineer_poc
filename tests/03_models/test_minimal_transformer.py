"""Section 3.3 / 3.4 / 3.5 — le modèle complet : pile de blocs, forward, lm_head.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/03_models/test_minimal_transformer.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(2 blocs, (1, 4, 16), (16, 8), ...) : c'est volontaire. Un test doit énoncer la vérité
attendue, pas la recalculer avec la même formule que le code testé.

Modèle jouet de la section 3, constant dans tout le fichier : `vocab_size=16`, `hidden=8`,
`num_layers=2`, `num_heads=2`, `seq=4`, `batch=1`. Ce fichier assemble ce qui existe déjà :
embedding (3.2) + N blocs transformer (2.13) + normalisation finale + lm_head. Aucune
nouveauté mathématique ici, uniquement la structure — et c'est cette structure que le
KV cache de la section 4 viendra instrumenter, couche par couche.

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
def test_minimal_transformer_stacks_requested_number_of_blocks():
    """Roadmap 3.3 — la profondeur du modèle est une donnée de config, pas du code dupliqué.

    Objectif d'apprentissage
    ------------------------
    Un LLM est une répétition du même bloc : Qwen2.5-0.5B empile 24 blocs identiques en
    structure, tous différents en poids. Deux règles en découlent, et elles comptent pour
    l'inférence : la boucle de forward itère sur une `ModuleList` (donc `num_layers` est
    un paramètre, jamais un copier-coller), et chaque bloc possède SES propres poids —
    c'est pourquoi le KV cache aura besoin d'un slot par couche (section 4.2) et pourquoi
    la mémoire des poids est proportionnelle à la profondeur (section 5.1).

    Schéma mental
    -------------
        token_ids (1, 4) -> embedding -> h0 (1, 4, 8)
                                          |
                            blocks[0] ----+-> h1 (1, 4, 8)     num_layers=2 blocs
                            blocks[1] ----+-> h2 (1, 4, 8)     shape INCHANGÉE à chaque étage
                                          v
                                    norm finale -> lm_head

        len(model.blocks) == 2      blocks[0] et blocks[1] : poids DISTINCTS

    Ce que ce test vérifie
    ----------------------
    1. le modèle expose ses blocs dans une `torch.nn.ModuleList` de longueur `num_layers = 2` ;
    2. cette longueur est pilotée par la config, et non fixée dans le code ;
    3. les deux blocs sont des objets distincts qui ne partagent AUCUN tenseur de paramètres
       (sinon on aurait un seul bloc appliqué deux fois, ce qui n'est pas un transformer à
       2 couches).

    API à faire émerger (cible roadmap : `src/inference_lab/models/minimal_transformer/model.py`)
    --------------------------------------------------------------------------------------------
        class MinimalTransformer(torch.nn.Module):
            def __init__(self, config: MinimalTransformerConfig) -> None: ...
            config: MinimalTransformerConfig
            blocks: torch.nn.ModuleList
            def forward(self, token_ids: torch.Tensor) -> torch.Tensor: ...

    Indice : `torch.nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))`
    — une `ModuleList` (pas une liste Python) est indispensable pour que `model.parameters()`
    voie les poids des blocs. Piège classique : construire UN bloc et le mettre `num_layers`
    fois dans la liste ; l'assert 3 est là précisément pour l'attraper.
    """

    pytest.skip("Roadmap TDD 3.3 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.models.minimal_transformer.model import MinimalTransformer

    # Arrange — fixer la graine (`torch.manual_seed(0)`) puis construire :
    #           - `config`, la config de la section 3.1 : vocab_size=16, hidden_size=8,
    #             num_layers=2, num_heads=2 ;
    #           - `model`, un `MinimalTransformer` construit à partir de cette config.

    # Act — rien à exécuter : c'est la STRUCTURE du modèle instancié que l'on inspecte,
    #       via `model.blocks` et les paramètres de chaque bloc.

    # Assert 1 — les blocs sont empilés dans une ModuleList de longueur num_layers
    assert isinstance(model, MinimalTransformer)
    assert isinstance(model.blocks, torch.nn.ModuleList)
    assert len(model.blocks) == 2

    # Assert 2 — la profondeur vient de la config
    assert len(model.blocks) == model.config.num_layers

    # Assert 3 — chaque bloc a ses propres poids
    assert model.blocks[0] is not model.blocks[1]
    assert all(
        first is not second
        for first, second in zip(
            model.blocks[0].parameters(), model.blocks[1].parameters(), strict=True
        )
    )


@pytest.mark.tdd
@pytest.mark.model
def test_model_forward_outputs_logits_for_each_token_and_vocabulary_entry():
    """Roadmap 3.4 — le forward rend un score par token du vocabulaire, pour chaque position.

    Objectif d'apprentissage
    ------------------------
    La sortie d'un LLM n'est pas « le mot suivant » : c'est un tenseur de logits
    `(batch, seq, vocab_size)`, soit une distribution non normalisée sur tout le vocabulaire
    à CHAQUE position. Cette forme explique tout le reste du projet : en prefill on calcule
    les `seq` lignes d'un coup, en decode on n'a besoin que de la DERNIÈRE (`logits[:, -1, :]`),
    et l'échantillonnage (3.6 à 3.9) travaille sur cette dernière ligne. Sur Qwen2.5-0.5B,
    cette dernière dimension vaut 151936 : c'est la plus grosse matmul du decode.

    Schéma mental
    -------------
        token_ids (batch=1, seq=4)   [entiers dans [0, 16)]
             |
             v  embedding + 2 blocs + norm
        hidden (1, 4, hidden=8)
             |
             v  lm_head (8 -> 16)
        logits (1, 4, vocab_size=16)   [float32, non normalisés]

        softmax(logits, dim=-1).sum(dim=-1) == 1 pour chacune des 1 x 4 positions

    Ce que ce test vérifie
    ----------------------
    1. les logits ont exactement la shape `(batch, seq, vocab_size) = (1, 4, 16)` ;
    2. batch et seq sont ceux de l'entrée : le modèle ne réduit ni ne réordonne les positions ;
    3. les logits sont des float32 tous finis (pas de `nan`, pas d'`inf`) — un `nan` ici
       signale presque toujours une normalisation ou un masque cassé ;
    4. ce sont bien des scores NON normalisés : leur softmax sur la dernière dimension somme
       à 1 pour chaque position, alors que les logits eux-mêmes n'ont aucune raison de le faire.

    API à faire émerger (cible roadmap : `src/inference_lab/models/minimal_transformer/model.py`)
    --------------------------------------------------------------------------------------------
        def forward(self, token_ids: torch.Tensor) -> torch.Tensor: ...
            # (batch, seq) entiers -> (batch, seq, vocab_size) float32

    Indice : enchaîne embedding, boucle `for block in self.blocks`, normalisation finale puis
    `self.lm_head`. Fais le forward sous `torch.no_grad()` dans le test si tu veux, mais ne
    renvoie JAMAIS un softmax depuis le modèle : la perte d'entraînement
    (`torch.nn.functional.cross_entropy`) comme l'échantillonnage attendent des logits bruts.
    """

    pytest.skip("Roadmap TDD 3.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.models.minimal_transformer.model import MinimalTransformer

    # Arrange — fixer la graine (`torch.manual_seed(0)`) puis construire :
    #           - `config` (vocab_size=16, hidden_size=8, num_layers=2, num_heads=2) et
    #             `model`, le `MinimalTransformer` correspondant ;
    #           - `token_ids`, un tenseur `torch.long` de shape (batch=1, seq=4) dont tous les
    #             ids sont dans [0, 16).

    # Act — appeler le forward du modèle sur `token_ids` pour obtenir `logits`.

    # Assert 1 — une ligne de scores par position, une colonne par entrée du vocabulaire
    assert isinstance(model, MinimalTransformer)
    assert logits.shape == (1, 4, 16)

    # Assert 2 — batch et seq sont ceux de l'entrée
    assert logits.shape[:2] == token_ids.shape
    assert logits.shape[-1] == 16

    # Assert 3 — des float32 tous finis
    assert logits.dtype is torch.float32
    assert bool(torch.isfinite(logits).all())

    # Assert 4 — ce sont des scores non normalisés, dont le softmax est une distribution
    probabilities = logits.softmax(dim=-1)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(1, 4), rtol=1e-5, atol=1e-6)


@pytest.mark.tdd
@pytest.mark.model
def test_lm_head_projects_hidden_dimension_to_vocabulary_size():
    """Roadmap 3.5 — le lm_head est la seule couche qui change d'espace : hidden -> vocabulaire.

    Objectif d'apprentissage
    ------------------------
    Tout le corps du transformer travaille en dimension `hidden` : chaque bloc entre et sort
    en `(batch, seq, hidden)`. Une seule projection finale, le `lm_head`, quitte cet espace
    pour celui du vocabulaire. Sa matrice a la forme `(vocab_size, hidden)` — exactement celle
    de la table d'embedding transposée, ce qui rend le weight tying possible. En decode, ce
    `Linear` est appliqué à UNE position mais avec `vocab_size` sorties : sur Qwen2.5-0.5B
    c'est 896 x 151936 ≈ 136 M paramètres lus à chaque token généré.

    Schéma mental
    -------------
        hidden_states (batch=1, seq=4, hidden=8)
             |
             v  lm_head : Linear(in_features=8, out_features=16)
        logits (1, 4, vocab_size=16)

        lm_head.weight.shape == (16, 8)          # (out_features, in_features)
        logits = hidden_states @ weight.T (+ bias éventuel)

    Ce que ce test vérifie
    ----------------------
    1. le `lm_head` est un `torch.nn.Linear` de `hidden = 8` vers `vocab_size = 16`, avec une
       matrice de poids `(16, 8)` ;
    2. appliqué à des états cachés `(1, 4, 8)`, il produit `(1, 4, 16)` : seule la dernière
       dimension change ;
    3. il calcule bien la projection linéaire de référence `x @ weight.T + bias`, oracle
       fourni par `torch.nn.functional.linear` ;
    4. ses dimensions sont dérivées de la config, pas écrites en dur dans le modèle.

    API à faire émerger (cible roadmap : `src/inference_lab/models/minimal_transformer/model.py`)
    --------------------------------------------------------------------------------------------
        class MinimalTransformer(torch.nn.Module):
            lm_head: torch.nn.Linear  # Linear(config.hidden_size, config.vocab_size)

    Indice : `torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)` suffit ; note
    que PyTorch stocke `weight` en `(out_features, in_features)`, donc `(16, 8)` et non `(8, 16)`
    — cette transposition implicite est la source d'erreur numéro un ici. `bias=False` est le
    choix des LLM récents (Llama, Qwen2) ; `torch.nn.functional.linear` accepte `bias=None`,
    l'assert 3 reste donc valable dans les deux cas.
    """

    pytest.skip("Roadmap TDD 3.5 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.models.minimal_transformer.model import MinimalTransformer

    # Arrange — fixer la graine (`torch.manual_seed(0)`) puis construire :
    #           - `config` (vocab_size=16, hidden_size=8, num_layers=2, num_heads=2) et
    #             `model`, le `MinimalTransformer` correspondant ;
    #           - `hidden_states`, un tenseur float32 aléatoire de shape
    #             (batch=1, seq=4, hidden=8) qui joue le rôle de la sortie du dernier bloc.

    # Act — projeter `hidden_states` avec `model.lm_head` pour obtenir `logits`.

    # Assert 1 — une projection Linear de hidden vers le vocabulaire
    assert isinstance(model, MinimalTransformer)
    assert isinstance(model.lm_head, torch.nn.Linear)
    assert model.lm_head.in_features == 8
    assert model.lm_head.out_features == 16
    assert model.lm_head.weight.shape == (16, 8)

    # Assert 2 — seule la dernière dimension change
    assert logits.shape == (1, 4, 16)

    # Assert 3 — c'est la projection linéaire de référence
    torch.testing.assert_close(
        logits,
        torch.nn.functional.linear(hidden_states, model.lm_head.weight, model.lm_head.bias),
        rtol=1e-5,
        atol=1e-6,
    )

    # Assert 4 — les dimensions viennent de la config
    assert model.lm_head.in_features == model.config.hidden_size
    assert model.lm_head.out_features == model.config.vocab_size
