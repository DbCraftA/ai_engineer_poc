"""Section 9.2 — vector add en Triton : le kernel minimal, comparé bit à bit à PyTorch.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/09_triton/test_vector_add.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(shapes (4096,) et (1000,), égalité bit à bit avec `x + y`) : c'est volontaire. Un test doit
énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Cap du chapitre : on n'écrit pas un kernel Triton pour « aller plus vite », on l'écrit pour
répondre à une question mémoire/kernel identifiée — ici la plus simple de toutes : lire deux
tenseurs, en écrire un troisième, sans jamais déborder. Ce test ne mesure donc AUCUNE
performance : il vérifie la CORRECTION du kernel, et l'oracle est PyTorch. `x + y` élément par
élément est la seule opération du chapitre où l'égalité bit à bit est légitime, ce qui en fait
le point de départ idéal avant la réduction de 9.3.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
@pytest.mark.triton
def test_triton_vector_add_matches_pytorch():
    """Roadmap 9.2 — une addition élémentaire en fp32 est reproductible au bit près.

    Objectif d'apprentissage
    ------------------------
    Un kernel Triton découpe le tenseur en blocs, un « program » par bloc, chacun calculant
    ses propres indices. Deux notions y sont installées une fois pour tout le chapitre. La
    première : le wrapper Python. Le kernel `@triton.jit` n'alloue rien et ne renvoie rien ;
    c'est une fonction hôte qui alloue la sortie, calcule la grille et lance le kernel — c'est
    elle que le test importe. La seconde : le masque. Quand la taille n'est pas un multiple de
    la taille de bloc, le dernier program adresse des offsets hors du tenseur ; sans
    `mask=offs < n`, on lit et surtout on ÉCRIT dans la mémoire du voisin, corruption
    silencieuse que seul un test comme celui-ci attrape.

    Point numérique central : une addition élémentaire ne réduit rien et ne réordonne rien.
    Chaque sortie est le résultat d'UNE seule addition fp32 entre les deux mêmes opérandes que
    PyTorch, et l'arrondi IEEE-754 d'une addition est déterministe. Le résultat est donc
    identique au bit près : `torch.equal` est ici légitime, alors qu'il serait faux dès qu'une
    somme sur plusieurs termes entre en jeu (9.3, 9.4).

    Schéma mental
    -------------
        n = 4096, BLOCK = 256  ->  grille de 16 programs, tous pleins
            program 3 : offs = 3*256 + [0..255] = [768..1023], masque tout vrai

        n = 1000, BLOCK = 256  ->  grille de 4 programs, 1000 = 3 x 256 + 232
            program 3 : offs = [768..1023], masque vrai sur 232 valeurs, faux sur 24
            sans masque : 24 écritures hors tenseur

        sortie[i] = x[i] + y[i] : une addition, aucune réduction  ->  bit à bit exact

    Ce que ce test vérifie
    ----------------------
    1. le wrapper alloue lui-même la sortie : shape (4096,), float32, sur `"cuda"`, contiguë ;
    2. le résultat est EXACTEMENT celui de `x + y` de PyTorch, au bit près ;
    3. avec n = 1000 et une taille de bloc de 256, le dernier bloc est incomplet et le
       masquage garde la queue du tenseur correcte ;
    4. le kernel travaille hors place : ses deux tenseurs d'entrée sont inchangés.

    API à faire émerger (cible roadmap : `src/inference_lab/kernels/triton/vector_add.py`)
    ------------------------------------------------------------------------------------
        def triton_vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

    Indice : dans le kernel, `pid = tl.program_id(axis=0)`,
    `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`, `mask = offs < n_elements`, puis
    `tl.load(x_ptr + offs, mask=mask)` et `tl.store(out_ptr + offs, x + y, mask=mask)`. Côté
    wrapper : `out = torch.empty_like(x)` et une grille
    `(triton.cdiv(n_elements, BLOCK_SIZE),)`. Pièges : oublier le masque (le test le voit sur
    n = 1000), passer une grille trop petite (la queue du tenseur reste à la valeur non
    initialisée de `torch.empty_like`), et accepter des entrées non contiguës alors que le
    kernel raisonne en offsets linéaires.
    """

    pytest.skip("Roadmap TDD 9.2 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.kernels.triton.vector_add import triton_vector_add

    # Arrange — appeler `torch.manual_seed(0)` d'abord, puis créer, tous en `torch.float32`,
    #           contigus et alloués directement sur `"cuda"` : `x` et `y` de shape (4096,),
    #           soit 16 blocs pleins de 256 éléments ; `x_ref` et `y_ref`, copies exactes de
    #           `x` et `y` (clones), pour vérifier que le kernel n'écrit pas dans ses entrées ;
    #           `x_tail` et `y_tail` de shape (1000,), taille volontairement NON multiple de
    #           256 (1000 = 3 x 256 + 232) pour forcer le masquage des accès mémoire.

    # Act — calculer `out = triton_vector_add(x, y)` puis
    #       `out_tail = triton_vector_add(x_tail, y_tail)`, le kernel utilisant une taille de
    #       bloc de 256 éléments.

    # Assert 1 — le wrapper alloue la sortie : shape, dtype, device et contiguïté
    assert out.shape == (4096,)
    assert out.dtype is torch.float32
    assert out.device.type == "cuda"
    assert out.is_contiguous()

    # Assert 2 — une addition élémentaire fp32 : égalité BIT À BIT avec l'oracle PyTorch
    assert torch.equal(out, x + y)
    assert torch.equal(triton_vector_add(x, y), x + y)

    # Assert 3 — dernier bloc incomplet : le masque protège les 232 dernières valeurs
    assert out_tail.shape == (1000,)
    assert torch.equal(out_tail, x_tail + y_tail)
    assert torch.equal(out_tail[768:], (x_tail + y_tail)[768:])

    # Assert 4 — calcul hors place : les entrées ne sont pas écrasées
    assert torch.equal(x, x_ref)
    assert torch.equal(y, y_ref)
