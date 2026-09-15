"""Section 2.13 — bloc transformer : deux sous-couches, deux normalisations, deux résidus.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/02_attention/test_transformer_block.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
((1, 4, 8) en entrée comme en sortie, sous-couches nulles -> sortie = entrée) : c'est
volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule que
le code testé.

Dimensions constantes de la section : batch=1, seq=4, hidden=8, num_heads=2, head_dim=4,
intermediate=16. Ce test ferme la section 2 : il assemble 2.1 à 2.12 en une brique empilable.
Qwen2.5-0.5B n'est rien d'autre que 24 exemplaires de ce bloc, ce qui explique pourquoi la
conservation de la shape est une contrainte absolue.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_transformer_block_preserves_hidden_shape():
    """Roadmap 2.13 — un bloc rend exactement la shape qu'il reçoit, donc il s'empile.

    Objectif d'apprentissage
    ------------------------
    Toute la profondeur d'un LLM repose sur une propriété simple : la sortie d'un bloc a la
    même forme que son entrée, ce qui permet d'en chaîner 24, 32 ou 80 sans aucune adaptation.
    À l'intérieur, les dimensions changent pourtant sans arrêt (têtes en 2.7, espace
    intermédiaire en 2.12) : le bloc est le niveau où l'on redevient un simple flux
    (batch, seq, hidden). C'est aussi ce contrat qui permettra de mesurer une couche isolément
    et d'extrapoler le coût du modèle complet.

    Schéma mental
    -------------
        x (batch=1, seq=4, hidden=8), num_heads=2, head_dim=4, intermediate=16

        x --norm--> attention --+--> h --norm--> mlp --+--> sortie (1, 4, 8)
          \\___________________/          \\___________/
                 résidu                       résidu

        bloc(bloc(x)) est légal : même shape en entrée et en sortie (24 fois pour Qwen2.5-0.5B)

    Ce que ce test vérifie
    ----------------------
    1. la sortie a exactement la shape d'entrée (1, 4, 8) ;
    2. le bloc s'applique à sa propre sortie : deux blocs chaînés gardent la shape ;
    3. dtype conservé et aucune valeur non finie (les normalisations font leur travail) ;
    4. la sortie n'est pas l'entrée : le bloc calcule réellement quelque chose.

    API à faire émerger (cible roadmap : `src/inference_lab/nn/transformer_block.py`)
    -------------------------------------------------------------------------------
        class TransformerBlock(torch.nn.Module):
            def __init__(
                self,
                hidden_size: int,
                num_heads: int,
                intermediate_size: int,
                eps: float = 1e-6,
            ) -> None: ...

            def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    Indice : réutilise les modules déjà spécifiés (2.1 à 2.12) plutôt que de tout réécrire.
    Fixe `torch.manual_seed(0)` avant de construire le bloc pour que l'initialisation soit
    reproductible. Piège : oublier la projection de sortie de l'attention casse la shape dès
    que num_heads x head_dim est réorganisé.
    """

    pytest.skip("Roadmap TDD 2.13 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.transformer_block import TransformerBlock

    # Arrange — batch=1, seq=4, hidden=8, num_heads=2, head_dim=4, intermediate=16.
    #           `block` : une instance de `TransformerBlock` construite avec hidden_size=8,
    #           num_heads=2, intermediate_size=16, initialisation reproductible (seed fixée) et
    #           poids non nuls.
    #           `x` : tenseur (1, 4, 8) en `torch.float32`, déterministe.

    # Act — calculer `out = block(x)`, puis `out_stacked` en appliquant `block` à `out`.

    # Assert 1 — contrat de base : la shape traverse le bloc sans changer
    assert isinstance(block, TransformerBlock)
    assert out.shape == (1, 4, 8)
    assert out.shape == x.shape

    # Assert 2 — donc le bloc est empilable tel quel
    assert out_stacked.shape == (1, 4, 8)

    # Assert 3 — dtype conservé, aucune valeur non finie
    assert out.dtype is torch.float32
    assert bool(out.isfinite().all())

    # Assert 4 — le bloc transforme vraiment ses entrées
    assert not torch.allclose(out, x)


@pytest.mark.tdd
def test_transformer_block_contains_attention_and_mlp_residual_paths():
    """Roadmap 2.13 — deux chemins résiduels indépendants traversent le bloc.

    Objectif d'apprentissage
    ------------------------
    Le bloc n'est pas une composition de fonctions, c'est une somme : `x + attention(norm(x))`
    puis `h + mlp(norm(h))`. Chaque sous-couche ne fait qu'AJOUTER une correction à un flux
    d'information qui traverse le bloc intact. D'où la façon la plus nette de le prouver : si
    les sous-couches ne produisent rien (poids de sortie nuls), le bloc doit rendre son entrée
    à l'identique. C'est ce chemin direct qui rend les modèles profonds entraînables, et c'est
    aussi lui qui impose que la sortie d'attention et la sortie de MLP aient la même dimension
    que `x`. En pré-norm (Llama, Qwen), la normalisation est à l'INTÉRIEUR de la branche, donc
    le résidu n'est jamais normalisé.

    Schéma mental
    -------------
        h      = x + attn_out(x)      avec attn_out = W_o @ attention(norm(x))
        sortie = h + mlp_out(h)       avec mlp_out  = W_down @ swiglu(norm(h))

        W_o = 0 et W_down = 0   ->  sortie == x, exactement (chemin résiduel pur)
        W_down = 0 seul         ->  sortie == x + attn_out(x) != x
        W_o = 0 seul            ->  sortie == x + mlp_out(x) != x

    Ce que ce test vérifie
    ----------------------
    1. avec les deux projections de sortie mises à zéro, la sortie est EXACTEMENT l'entrée ;
    2. avec seule la voie MLP neutralisée, la contribution de l'attention subsiste ;
    3. avec seule la voie attention neutralisée, la contribution du MLP subsiste ;
    4. le bloc expose la structure pré-norm attendue : deux normalisations, une attention,
       un MLP.

    API à faire émerger (cible roadmap : `src/inference_lab/nn/transformer_block.py`)
    -------------------------------------------------------------------------------
        class TransformerBlock(torch.nn.Module):
            input_layernorm: torch.nn.Module
            self_attn: torch.nn.Module
            post_attention_layernorm: torch.nn.Module
            mlp: torch.nn.Module

            def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    Indice : ces quatre noms d'attributs sont ceux de `Qwen2DecoderLayer` chez Hugging Face ;
    les reprendre rendra le chargement des poids trivial en section 5. Pour neutraliser une
    branche, mets à zéro `self_attn.o_proj.weight` et/ou `mlp.down_proj.weight` sous
    `torch.no_grad()` (et les biais éventuels). Piège : si tu écris `x = norm(x)` avant le
    résidu, l'assert 1 casse — le résidu doit partir du `x` NON normalisé.
    """

    pytest.skip("Roadmap TDD 2.13 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.transformer_block import TransformerBlock

    # Arrange — batch=1, seq=4, hidden=8, num_heads=2, head_dim=4, intermediate=16.
    #           `x` : tenseur (1, 4, 8) en `torch.float32`, déterministe (seed fixée).
    #           `block_zeroed` : un `TransformerBlock` (hidden_size=8, num_heads=2,
    #           intermediate_size=16) dont la projection de sortie de l'attention ET la
    #           projection descendante du MLP sont mises à zéro, biais compris, sous
    #           `torch.no_grad()`.
    #           `block_attn_only` : le même bloc avec SEULE la projection descendante du MLP
    #           à zéro.
    #           `block_mlp_only` : le même bloc avec SEULE la projection de sortie de
    #           l'attention à zéro.

    # Act — appliquer les trois blocs à `x` pour obtenir `out_zeroed`, `out_attn_only` et
    #       `out_mlp_only`.

    # Assert 1 — sous-couches muettes : le bloc se réduit au chemin résiduel, sortie == entrée
    torch.testing.assert_close(out_zeroed, x, atol=1e-6, rtol=0.0)

    # Assert 2 — la branche attention contribue bien quand elle est active
    assert not torch.allclose(out_attn_only, x)

    # Assert 3 — la branche MLP contribue bien quand elle est active
    assert not torch.allclose(out_mlp_only, x)

    # Assert 4 — structure pré-norm exposée, avec les noms de la référence Qwen2
    assert isinstance(block_zeroed, TransformerBlock)
    assert isinstance(block_zeroed, torch.nn.Module)
    assert hasattr(block_zeroed, "input_layernorm")
    assert hasattr(block_zeroed, "self_attn")
    assert hasattr(block_zeroed, "post_attention_layernorm")
    assert hasattr(block_zeroed, "mlp")
