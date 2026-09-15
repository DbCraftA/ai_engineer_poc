"""Section 2.10 — RoPE : encoder la position par une rotation de Q et K.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/02_attention/test_rope.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(norme inchangée, position 0 = identité, scores égaux à écart de positions égal) : c'est
volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule
que le code testé.

Dimensions constantes de la section : batch=1, seq=4, hidden=8, num_heads=2, head_dim=4.
RoPE s'applique à Q et K juste avant les scores de 2.2, jamais à V. C'est aussi la raison pour
laquelle le KV cache stocke des K DÉJÀ tournés : la position est gravée dans le tenseur, elle
n'est pas ajoutée à l'entrée comme un embedding appris.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_rope_preserves_vector_norm():
    """Roadmap 2.10 — une rotation change la direction, jamais la longueur.

    Objectif d'apprentissage
    ------------------------
    RoPE ne fait pas qu'« ajouter » une position : il fait tourner chaque paire de
    coordonnées d'une tête d'un angle proportionnel à la position. Comme toute rotation est
    une transformation orthogonale, la norme de chaque vecteur de tête est conservée. C'est
    cette propriété qui rend RoPE inoffensif pour la stabilité numérique : les magnitudes
    entrant dans le softmax ne dérivent pas quand la position augmente, contrairement à ce
    qu'un encodage additif mal calibré provoquerait sur un contexte long.

    Schéma mental
    -------------
        x (batch=1, heads=2, seq=4, head_dim=4), positions = [0, 1, 2, 3]

        paires de coordonnées : (x0, x1) et (x2, x3)
        angle de la paire j à la position m : m * theta_j, avec theta_j = 10000^(-2j/head_dim)

            [x0, x1] --rotation d'angle m*theta_0--> [x0', x1']
            sqrt(x0'^2 + x1'^2) == sqrt(x0^2 + x1^2)

        donc ||x_rot[b, h, m, :]|| == ||x[b, h, m, :]||  pour toute position m
        position 0 : angle nul, cos(0)=1 et sin(0)=0  ->  vecteur inchangé

    Ce que ce test vérifie
    ----------------------
    1. shape et dtype sont conservés : (1, 2, 4, 4) en float32 ;
    2. la norme de CHAQUE vecteur de tête est inchangée (rotation orthogonale) ;
    3. la position 0 laisse le vecteur strictement identique ;
    4. les autres positions modifient réellement le vecteur (ce n'est pas l'identité partout).

    API à faire émerger (cible roadmap : `src/inference_lab/nn/positional/rope.py`)
    -----------------------------------------------------------------------------
        def apply_rope(
            x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0
        ) -> torch.Tensor: ...

    Indice : `inv_freq = theta ** (-torch.arange(0, head_dim, 2).float() / head_dim)` donne un
    angle par paire, `angles = positions[:, None] * inv_freq` les angles de chaque position,
    puis `cos`/`sin` se combinent aux paires de coordonnées. Piège : `head_dim` doit être
    PAIR, et les deux conventions de découpage en paires (voisines `(x0, x1)` ou moitiés
    `(x0, x2)`) donnent des résultats différents — choisis-en une et documente-la, la
    compatibilité avec les poids Hugging Face en dépend.
    """

    pytest.skip("Roadmap TDD 2.10 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.positional.rope import apply_rope

    # Arrange — batch=1, num_heads=2, seq=4, head_dim=4 (pair, condition de RoPE).
    #           `x` : tenseur (1, 2, 4, 4) en `torch.float32`, déterministe (seed fixée),
    #           sans vecteur nul (une rotation d'un vecteur nul ne prouverait rien).
    #           `positions` : tenseur d'entiers 1D de longueur 4 valant les positions
    #           successives 0, 1, 2, 3.

    # Act — calculer `rotated = apply_rope(x, positions)`.

    # Assert 1 — RoPE est une transformation sur place au sens des shapes
    assert rotated.shape == (1, 2, 4, 4)
    assert apply_rope(x, positions).shape == x.shape
    assert rotated.dtype is torch.float32

    # Assert 2 — rotation = norme conservée, vecteur de tête par vecteur de tête
    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-6, rtol=1e-5)

    # Assert 3 — à la position 0, l'angle est nul : le vecteur est inchangé
    torch.testing.assert_close(rotated[:, :, 0], x[:, :, 0], atol=1e-6, rtol=1e-6)

    # Assert 4 — aux positions suivantes, la rotation agit vraiment
    assert not torch.allclose(rotated[:, :, 1], x[:, :, 1])
    assert not torch.allclose(rotated[:, :, 3], x[:, :, 3])


@pytest.mark.tdd
def test_rope_changes_representation_according_to_position():
    """Roadmap 2.10 — même vecteur, positions différentes : représentations différentes.

    Objectif d'apprentissage
    ------------------------
    Sans information de position, l'attention est invariante par permutation des tokens :
    « le chat mange » et « mange le chat » donneraient les mêmes scores. RoPE brise cette
    symétrie en tournant le même vecteur d'un angle différent selon sa position. Sa propriété
    remarquable est plus forte : le produit scalaire entre une requête en position m et une
    clé en position n ne dépend QUE de l'écart m - n. L'attention devient donc relative, ce
    qui explique la capacité à extrapoler au-delà de la longueur vue en entraînement et
    l'existence de recettes comme le scaling de `theta`.

    Schéma mental
    -------------
        même vecteur v répété aux 4 positions : x[0, 0, m, :] == v pour m = 0, 1, 2, 3

        après RoPE :
            R_0 v == v            (identité)
            R_1 v != R_2 v        (positions différentes -> vecteurs différents)

        produit scalaire (matrice 4 x 4 des scores de ces vecteurs tournés) :
            <R_0 v, R_1 v> == <R_1 v, R_2 v> == <R_2 v, R_3 v>     (écart de 1)
        seul l'ÉCART de positions compte, pas les positions absolues

    Ce que ce test vérifie
    ----------------------
    1. la position 0 ne change rien, et deux positions différentes donnent deux vecteurs
       différents pour la même entrée ;
    2. les scores ne dépendent que de l'écart de positions (propriété relative de RoPE) ;
    3. les scores de la diagonale (écart nul) sont tous égaux entre eux ;
    4. malgré ces changements, toutes les normes restent celles du vecteur d'origine.

    API à faire émerger (cible roadmap : `src/inference_lab/nn/positional/rope.py`)
    -----------------------------------------------------------------------------
        def apply_rope(
            x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0
        ) -> torch.Tensor: ...

    Indice : la matrice des scores s'obtient avec `rotated[0, 0] @ rotated[0, 0].T` ;
    l'assert 2 compare `scores[0, 1]`, `scores[1, 2]` et `scores[2, 3]`, tous d'écart 1.
    Piège : si tu appliques la même position à tous les tokens (ou si tu oublies d'utiliser
    `positions`), l'assert 1 casse ; si tu tournes les coordonnées indépendamment au lieu de
    les tourner par paires, c'est l'assert 2 qui casse.
    """

    pytest.skip("Roadmap TDD 2.10 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.positional.rope import apply_rope

    # Arrange — batch=1, num_heads=2, seq=4, head_dim=4.
    #           `x_same` : tenseur (1, 2, 4, 4) en `torch.float32` où le MÊME vecteur de
    #           head_dim=4 valeurs non nulles et distinctes est répété aux 4 positions ; ainsi
    #           seule la position varie d'un token à l'autre.
    #           `positions` : tenseur d'entiers 1D valant 0, 1, 2, 3.

    # Act — calculer `rotated = apply_rope(x_same, positions)`, puis la matrice de scores
    #       `scores = rotated[0, 0] @ rotated[0, 0].T` de shape (4, 4).

    # Assert 1 — position 0 : identité ; positions différentes : vecteurs différents
    torch.testing.assert_close(rotated[0, 0, 0], x_same[0, 0, 0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        apply_rope(x_same, positions)[0, 0, 0], x_same[0, 0, 0], atol=1e-6, rtol=1e-6
    )
    assert not torch.allclose(rotated[0, 0, 1], rotated[0, 0, 2])
    assert not torch.allclose(rotated[0, 0, 0], rotated[0, 0, 1])

    # Assert 2 — seul l'écart de positions compte : écart 1 partout, donc même score
    torch.testing.assert_close(scores[0, 1], scores[1, 2], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(scores[1, 2], scores[2, 3], atol=1e-6, rtol=1e-5)

    # Assert 3 — écart nul : la diagonale est constante
    torch.testing.assert_close(scores[0, 0], scores[3, 3], atol=1e-6, rtol=1e-5)

    # Assert 4 — les normes restent celles du vecteur d'origine
    torch.testing.assert_close(
        rotated[0, 0].norm(dim=-1), x_same[0, 0].norm(dim=-1), atol=1e-6, rtol=1e-5
    )
