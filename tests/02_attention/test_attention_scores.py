"""Section 2.2 / 2.3 — scores d'attention Q @ K^T et mise à l'échelle par 1/sqrt(head_dim).

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/02_attention/test_attention_scores.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
((1, 2, 4, 4), facteur 0.5 = 1/sqrt(4), ...) : c'est volontaire. Un test doit énoncer la
vérité attendue, pas la recalculer avec la même formule que le code testé.

Dimensions constantes de la section : batch=1, seq=4, hidden=8, num_heads=2, head_dim=4.
Les scores sont la matrice (seq, seq) qui coûte le plus cher en attention naïve : sa taille
croît en O(seq^2), ce qui motivera plus tard FlashAttention et l'attention en ligne.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_attention_scores_compare_every_query_with_every_key():
    """Roadmap 2.2 — un score est le produit scalaire d'une requête avec une clé.

    Objectif d'apprentissage
    ------------------------
    L'attention mesure une similarité : pour chaque token requête, on calcule un produit
    scalaire avec chaque token clé. Toutes ces similarités tiennent dans une seule matrice
    `Q @ K^T` de forme (seq_q, seq_k), une par tête. La dimension `head_dim` est contractée
    (elle disparaît) et remplacée par le nombre de clés. C'est cette matrice en seq x seq
    qui explique pourquoi le prefill coûte cher quand le prompt s'allonge.

    Schéma mental
    -------------
        Q (batch=1, heads=2, seq=4, head_dim=4)
        K (1, 2, 4, 4)  --transpose(-2, -1)-->  K^T (1, 2, 4, 4)

        scores = Q @ K^T  ->  (1, 2, 4, 4) = (batch, heads, seq_q=4, seq_k=4)
        ici seq et head_dim valent tous les deux 4 : la shape ne suffit pas à le voir,
        c'est `scores.shape[-1] == nombre de clés` qui porte le sens.

        Cas repère : si chaque tête de Q et de K contient la base canonique de R^4
        (e_0, e_1, e_2, e_3), alors scores = identité(4) : 1 sur la diagonale, 0 ailleurs.

    Ce que ce test vérifie
    ----------------------
    1. la shape des scores est (batch, heads, seq_q, seq_k) = (1, 2, 4, 4), la dernière
       dimension étant le nombre de clés et non `head_dim` ;
    2. sur des vecteurs orthonormés, les scores valent exactement l'identité ;
    3. quand Q et K sont le même tenseur, la matrice de scores est symétrique ;
    4. chaque tête est calculée indépendamment des autres.

    API à faire émerger (cible roadmap : `src/inference_lab/nn/attention/naive.py`)
    -----------------------------------------------------------------------------
        def attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor: ...

    Indice : `q @ k.transpose(-2, -1)` (ou `torch.matmul`) diffuse automatiquement sur les
    dimensions batch et têtes. Piège : `k.T` est interdit au-delà de 2 dimensions, et
    transposer les deux mauvaises dimensions donne une shape plausible mais un résultat faux.
    """

    pytest.skip("Roadmap TDD 2.2 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.attention.naive import attention_scores

    # Arrange — batch=1, num_heads=2, seq=4, head_dim=4.
    #           `q` et `k` : deux tenseurs (1, 2, 4, 4) en `torch.float32`, différents l'un de
    #           l'autre, valeurs déterministes (seed fixée) et non constantes.
    #           `q_basis` et `k_basis` : deux tenseurs (1, 2, 4, 4) dont CHAQUE tête contient
    #           la base canonique de R^4, c'est-à-dire quatre vecteurs unitaires orthogonaux.

    # Act — calculer `scores = attention_scores(q, k)`, `scores_basis` sur les deux tenseurs
    #       de base canonique, et `scores_self` en passant `q` comme requêtes ET comme clés.

    # Assert 1 — (batch, heads, seq_q, seq_k) : la dernière dimension compte les clés
    assert scores.shape == (1, 2, 4, 4)
    assert scores.shape[-1] == k.shape[-2]

    # Assert 2 — vecteurs orthonormés : 1 sur la diagonale, 0 partout ailleurs
    torch.testing.assert_close(scores_basis, torch.eye(4).expand(1, 2, 4, 4), atol=1e-6, rtol=0.0)

    # Assert 3 — Q contre lui-même : le produit scalaire est symétrique
    torch.testing.assert_close(scores_self, scores_self.transpose(-2, -1), atol=1e-6, rtol=0.0)

    # Assert 4 — chaque tête est indépendante : la tête 0 seule donne les mêmes scores
    torch.testing.assert_close(
        attention_scores(q[:, 0:1], k[:, 0:1])[0, 0], scores[0, 0], atol=1e-6, rtol=1e-6
    )


@pytest.mark.tdd
def test_attention_scores_are_scaled_by_inverse_sqrt_head_dimension():
    """Roadmap 2.3 — diviser par sqrt(head_dim) empêche le softmax de saturer.

    Objectif d'apprentissage
    ------------------------
    Un produit scalaire somme `head_dim` termes : plus la tête est large, plus les scores
    s'étalent. Sans correction, le softmax reçoit des logits énormes, sature en quasi
    one-hot et ne transmet presque plus de gradient ni d'information. Le facteur
    1/sqrt(head_dim) est exactement l'écart-type attendu de cette somme : il ramène les
    scores à une échelle stable, indépendante de `head_dim`. Qwen2.5-0.5B utilise
    head_dim=64, donc un facteur 1/8 ; ici head_dim=4, donc 1/2.

    Schéma mental
    -------------
        head_dim = 4  ->  1/sqrt(4) = 0.5

        scores_bruts    (1, 2, 4, 4)  --x 0.5-->  scores_scaled (1, 2, 4, 4)
        base canonique  diagonale 1.0  --x 0.5-->  diagonale 0.5

        softmax(scores_bruts) est plus piqué que softmax(scores_scaled) :
        diviser les logits aplatit la distribution (même effet qu'une température > 1).

    Ce que ce test vérifie
    ----------------------
    1. la mise à l'échelle ne change pas la shape (1, 2, 4, 4) ;
    2. les scores mis à l'échelle valent exactement les scores bruts x 0.5 ;
    3. sur des vecteurs orthonormés, la diagonale vaut exactement 0.5 ;
    4. conséquence : après softmax, la distribution mise à l'échelle est moins piquée que
       la distribution brute.

    API à faire émerger (cible roadmap : `src/inference_lab/nn/attention/naive.py`)
    -----------------------------------------------------------------------------
        def scaled_attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor: ...

    Indice : `head_dim` se lit sur `q.shape[-1]`, pas en paramètre ; utilise
    `math.sqrt` ou `head_dim**-0.5`. Piège classique : diviser par `seq` ou par
    `hidden` au lieu de `head_dim`, ce qui reste plausible en shape mais faux en valeur.
    """

    pytest.skip("Roadmap TDD 2.3 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.attention.naive import attention_scores, scaled_attention_scores

    # Arrange — batch=1, num_heads=2, seq=4, head_dim=4 (donc facteur attendu 1/sqrt(4) = 0.5).
    #           `q` et `k` : deux tenseurs (1, 2, 4, 4) en `torch.float32`, déterministes
    #           (seed fixée), dont les produits scalaires ne sont pas tous égaux — sinon le
    #           dernier assert compare deux distributions uniformes.
    #           `q_basis` et `k_basis` : deux tenseurs (1, 2, 4, 4) dont chaque tête contient
    #           la base canonique de R^4.

    # Act — calculer `raw = attention_scores(q, k)`, `scaled = scaled_attention_scores(q, k)`
    #       et `scaled_basis = scaled_attention_scores(q_basis, k_basis)`.

    # Assert 1 — la mise à l'échelle est élément par élément : shape inchangée
    assert scaled.shape == (1, 2, 4, 4)

    # Assert 2 — le facteur attendu, écrit en dur : 1/sqrt(4) = 0.5
    torch.testing.assert_close(scaled, raw * 0.5, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        scaled_attention_scores(q, k), attention_scores(q, k) * 0.5, atol=1e-6, rtol=0.0
    )

    # Assert 3 — sur des vecteurs orthonormés, la diagonale vaut exactement 0.5
    torch.testing.assert_close(scaled_basis[0, 0], 0.5 * torch.eye(4), atol=1e-6, rtol=0.0)

    # Assert 4 — des logits divisés donnent un softmax moins piqué
    assert torch.softmax(scaled, dim=-1).max() < torch.softmax(raw, dim=-1).max()
