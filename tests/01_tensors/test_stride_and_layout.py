"""Section 1.4 / 1.5 / 1.6 — strides, transposition et matérialisation d'un layout.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/01_tensors/test_stride_and_layout.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
((12, 4, 1), offset 23, (1, 3)) : c'est volontaire. Un test doit énoncer la vérité
attendue, pas la recalculer avec la même formule que le code testé.

La section 1.3 a montré qu'un tenseur est une *vue* sur un storage plat. Les strides sont
le chaînon manquant : ils disent de combien de cases avancer dans ce storage quand on
avance d'un cran sur un axe logique. Tout le reste en découle — pourquoi `transpose` est
gratuit, pourquoi `view` échoue parfois, et pourquoi un kernel réclame un tenseur contigu.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_contiguous_tensor_has_expected_strides():
    """Roadmap 1.4 — les strides d'un tenseur contigu sont les produits suffixes de la shape.

    Objectif d'apprentissage
    ------------------------
    Un tenseur contigu en *row-major* range le dernier axe en premier dans la mémoire :
    les éléments voisins sur le dernier axe sont voisins en RAM. Le stride d'un axe est
    donc le nombre d'éléments qu'occupe une « tranche » de tous les axes situés à sa droite.
    C'est la disposition que réclament les kernels rapides : lire `hidden[b, t]` d'un coup,
    c'est lire 8 valeurs consécutives, donc une seule ligne de cache. Savoir prédire les
    strides sans lancer PyTorch permet ensuite de raisonner sur la coalescence des accès
    GPU, sujet central des sections kernels.

    Schéma mental
    -------------
        shape (batch=2, seq=3, hidden=4), float32 contigu

            avancer de 1 sur hidden -> 1 élément plus loin
            avancer de 1 sur seq    -> 4 éléments plus loin   (une ligne de hidden)
            avancer de 1 sur batch  -> 12 éléments plus loin  (3 x 4, toute la séquence)

            strides = (12, 4, 1)         et pour (1, 4, 8) : (32, 8, 1)

    Ce que ce test vérifie
    ----------------------
    1. PyTorch donne bien `(12, 4, 1)` pour un tenseur contigu de shape `(2, 3, 4)` ;
    2. l'API prédit ces strides à partir de la seule shape, sans allouer de tenseur ;
    3. elle prédit aussi `(32, 8, 1)` pour la shape `(1, 4, 8)` des hidden states ;
    4. propriété structurelle : le dernier axe a toujours un stride de 1 en row-major.

    API à faire émerger (cible roadmap : `src/inference_lab/tensors/layout.py`)
    -------------------------------------------------------------------------
        def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]: ...

    Indice : parcours la shape de droite à gauche en accumulant le produit
    (`stride[-1] = 1`, `stride[i] = stride[i + 1] * shape[i + 1]`). Renvoie un `tuple`,
    pas une liste : `tensor.stride()` renvoie un tuple et la comparaison doit être directe.
    Piège : les strides comptent des ÉLÉMENTS, pas des octets — le dtype n'intervient pas.
    """

    pytest.skip("Roadmap TDD 1.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.tensors.layout import contiguous_strides

    # Arrange — créer `x`, un tenseur contigu de shape (2, 3, 4) en `torch.float32`
    #           (contenu indifférent : seul le layout est observé ici).

    # Act — demander à l'API les strides théoriques de la shape de `x`, dans `predicted`.

    # Assert 1 — la vérité PyTorch pour un tenseur contigu (2, 3, 4)
    assert x.is_contiguous()
    assert x.stride() == (12, 4, 1)

    # Assert 2 — l'API prédit la même chose à partir de la shape seule
    assert predicted == (12, 4, 1)
    assert predicted == x.stride()

    # Assert 3 — la shape des hidden states du projet, prédite sans allocation
    assert contiguous_strides((1, 4, 8)) == (32, 8, 1)

    # Assert 4 — en row-major, le dernier axe est toujours celui du pas unitaire
    assert contiguous_strides((2, 3, 4))[-1] == 1
    assert contiguous_strides((5,)) == (1,)


@pytest.mark.tdd
def test_stride_maps_indices_to_storage_offsets():
    """Roadmap 1.4 — un index multidimensionnel devient une adresse plate par produit scalaire.

    Objectif d'apprentissage
    ------------------------
    C'est la formule que le matériel exécute à chaque accès mémoire :

        offset = somme sur i de (index[i] * stride[i])

    Il n'existe pas de « tableau à 3 dimensions » en mémoire : il n'y a qu'un bloc plat et
    cette somme. C'est le calcul que tu écriras à la main dans un kernel Triton pour aller
    chercher la clé du token `t` de la tête `h` dans le KV cache, et c'est aussi ce qui
    explique pourquoi deux tenseurs de strides différents ont des performances différentes
    à shape identique.

    Schéma mental
    -------------
        x = arange(24) vu en (2, 3, 4), float32 contigu -> strides (12, 4, 1)
        la valeur stockée est égale à son offset, ce qui donne un oracle gratuit.

            index (0, 0, 0) -> 0*12 + 0*4 + 0*1 =  0   -> x[0, 0, 0] == 0.0
            index (0, 1, 0) -> 0*12 + 1*4 + 0*1 =  4   -> x[0, 1, 0] == 4.0
            index (1, 2, 3) -> 1*12 + 2*4 + 3*1 = 23   -> x[1, 2, 3] == 23.0

    Ce que ce test vérifie
    ----------------------
    1. le premier élément est à l'offset 0, et avancer d'un cran sur l'axe `seq` avance
       de 4 cases (la taille du dernier axe) ;
    2. le dernier élément du tenseur est à l'offset 23, calculé à la main ;
    3. l'offset prédit est bien celui où PyTorch a rangé la valeur : comme le tenseur
       contient `arange`, la valeur lue doit être égale à l'offset ;
    4. le contrat d'erreur : un index qui n'a pas autant de composantes que de strides
       est refusé par une `ValueError`.

    API à faire émerger (cible roadmap : `src/inference_lab/tensors/layout.py`)
    -------------------------------------------------------------------------
        def storage_offset(strides: tuple[int, ...], index: tuple[int, ...]) -> int: ...

    Indice : `sum(i * s for i, s in zip(index, strides, strict=True))` — le `strict=True`
    te donne gratuitement une partie du contrôle de l'assert 4. Piège : cette fonction
    prend les strides en paramètre, pas un tenseur, justement pour rester utilisable sur
    des layouts qui n'existent pas encore (un cache que l'on dimensionne sur papier).
    """

    pytest.skip("Roadmap TDD 1.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.tensors.layout import storage_offset

    # Arrange — créer `x`, un tenseur contigu de shape (2, 3, 4) en `torch.float32` dont les
    #           24 valeurs sont les entiers croissants à partir de 0 (via `torch.arange` puis
    #           un changement de shape), et `strides`, les strides réels de `x`.

    # Act — demander à l'API l'offset du premier élément, celui de l'index (0, 1, 0) et celui
    #       du dernier élément (1, 2, 3).

    # Assert 1 — origine et pas d'un cran sur l'axe des tokens
    assert strides == (12, 4, 1)
    assert storage_offset(strides, (0, 0, 0)) == 0
    assert storage_offset(strides, (0, 1, 0)) == 4

    # Assert 2 — le dernier élément logique, calculé à la main : 12 + 8 + 3
    assert storage_offset(strides, (1, 2, 3)) == 23

    # Assert 3 — l'oracle : dans un tenseur `arange`, la valeur EST son offset
    assert x[0, 1, 0].item() == 4.0
    assert x[1, 2, 3].item() == 23.0
    assert x.flatten()[storage_offset(strides, (1, 2, 3))].item() == 23.0

    # Assert 4 — un index de rang incohérent est une erreur, pas un offset approximatif
    with pytest.raises(ValueError):
        storage_offset(strides, (1, 2))


@pytest.mark.tdd
def test_transpose_changes_strides_without_reordering_storage():
    """Roadmap 1.5 — transposer, c'est échanger deux strides : aucune donnée ne bouge.

    Objectif d'apprentissage
    ------------------------
    `transpose` est une opération à coût nul : elle fabrique une nouvelle vue avec deux
    strides permutés, sur le même storage. C'est pour cela qu'un bloc d'attention peut
    passer de `[B, T, heads, head_dim]` à `[B, heads, T, head_dim]` sans payer de copie,
    et que le produit `Q @ K^T` s'écrit sans jamais matérialiser la transposée de K. La
    contrepartie arrive tout de suite : la vue obtenue n'est plus contiguë, et le
    parcours du dernier axe saute désormais dans la mémoire — d'où la section 1.6.

    Schéma mental
    -------------
        x  = arange(6) vu en (2, 3), float32   storage : [ 0 1 2 3 4 5 ]  24 octets
             shape (2, 3), strides (3, 1)

        xt = x.transpose(0, 1)
             shape (3, 2), strides (1, 3)      MÊME storage, mêmes 24 octets

             xt[0, 1] -> offset 0*1 + 1*3 = 3  -> storage[3] == 3.0
             xt n'est pas contigu : (1, 3) n'est pas le layout row-major de (3, 2)

    Ce que ce test vérifie
    ----------------------
    1. la transposée a la shape inversée `(3, 2)` et les strides échangés `(1, 3)` ;
    2. l'API prédit cet échange à partir des strides d'origine, sans tenseur ;
    3. le storage est intact : même adresse, même taille, mêmes valeurs dans le même
       ordre physique ;
    4. la vue transposée n'est plus contiguë, et lire `xt[0, 1]` retombe sur l'offset 3.

    API à faire émerger (cible roadmap : `src/inference_lab/tensors/layout.py`)
    -------------------------------------------------------------------------
        def swap_strides(strides: tuple[int, ...], dim0: int, dim1: int) -> tuple[int, ...]: ...

    Indice : convertis en liste, échange les deux positions, retourne un tuple. Piège :
    `transpose` ne « range » rien — après un `transpose`, `x.flatten()` reste une vue
    gratuite alors que `xt.flatten()` doit copier, puisque l'ordre logique ne suit plus
    l'ordre mémoire.
    """

    pytest.skip("Roadmap TDD 1.5 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.tensors.layout import swap_strides

    # Arrange — créer `x`, un tenseur contigu de shape (2, 3) en `torch.float32` dont les
    #           6 valeurs sont les entiers croissants à partir de 0, puis `xt`, la vue
    #           transposée de `x` obtenue en échangeant les axes 0 et 1.

    # Act — demander à l'API les strides attendus après échange des axes 0 et 1, dans
    #       `predicted`.

    # Assert 1 — la transposée : shape inversée, strides échangés
    assert xt.shape == torch.Size([3, 2])
    assert xt.stride() == (1, 3)

    # Assert 2 — l'API prédit l'échange à partir des strides d'origine
    assert predicted == (1, 3)
    assert swap_strides((32, 8, 1), 1, 2) == (32, 1, 8)

    # Assert 3 — le storage n'a pas été touché : même adresse, même taille, même ordre
    assert xt.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert xt.untyped_storage().nbytes() == 24
    assert x.flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    # Assert 4 — le prix à payer : la vue transposée n'est plus contiguë
    assert xt.is_contiguous() is False
    assert xt[0, 1].item() == 3.0


@pytest.mark.tdd
def test_contiguous_materializes_transposed_layout():
    """Roadmap 1.6 — `contiguous()` paie enfin la copie et réordonne physiquement les octets.

    Objectif d'apprentissage
    ------------------------
    Beaucoup de kernels (GEMM optimisés, `view`, kernels Triton écrits à la main) exigent
    un tenseur contigu : ils supposent que le dernier axe est un pas unitaire. Quand ce
    n'est pas le cas, il faut matérialiser le layout — c'est-à-dire allouer et recopier.
    Cette copie est un coût mémoire ET un coût de bande passante bien réel : dans un moteur
    d'inférence, on cherche à la faire une fois (au chargement des poids, en pré-transposant
    une matrice) plutôt qu'à chaque token décodé. Savoir qu'elle a lieu est la moitié du
    travail d'optimisation.

    Schéma mental
    -------------
        x           shape (2, 3), strides (3, 1)   storage A : [ 0 1 2 3 4 5 ]
        transposed  shape (3, 2), strides (1, 3)   storage A  (partagé, non contigu)
        materialized = transposed.contiguous()
                    shape (3, 2), strides (2, 1)   storage B : [ 0 3 1 4 2 5 ]
                                                               ^ octets réordonnés

        Les valeurs logiques sont identiques, l'ordre physique ne l'est plus.

    Ce que ce test vérifie
    ----------------------
    1. la vue transposée n'est pas contiguë, la version matérialisée l'est ;
    2. matérialiser alloue un nouveau storage (adresse différente) et rétablit les
       strides canoniques `(2, 1)` de la shape `(3, 2)` ;
    3. le contenu logique est préservé, mais l'ordre physique du storage devient
       `[0, 3, 1, 4, 2, 5]` : c'est là que se voit la copie ;
    4. sur un tenseur DÉJÀ contigu, l'appel ne copie rien : le storage est le même.

    API à faire émerger (cible roadmap : `src/inference_lab/tensors/layout.py`)
    -------------------------------------------------------------------------
        def ensure_contiguous(tensor: torch.Tensor) -> torch.Tensor: ...

    Indice : `tensor.contiguous()` fait déjà exactement cela, y compris le cas « déjà
    contigu » où il se contente de renvoyer le tenseur d'origine. L'intérêt du module est
    de nommer l'intention et de pouvoir y ajouter plus tard un compteur de copies.
    Piège : `materialized.flatten()` reflète l'ordre physique parce que le tenseur est
    contigu ; sur `transposed`, `flatten()` aurait dû copier pour répondre.
    """

    pytest.skip("Roadmap TDD 1.6 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.tensors.layout import ensure_contiguous

    # Arrange — créer `x`, un tenseur contigu de shape (2, 3) en `torch.float32` dont les
    #           6 valeurs sont les entiers croissants à partir de 0, puis `transposed`, sa
    #           vue transposée (axes 0 et 1 échangés).

    # Act — matérialiser `transposed` dans `materialized` via l'API.

    # Assert 1 — l'état de départ et l'état d'arrivée
    assert transposed.is_contiguous() is False
    assert materialized.is_contiguous() is True

    # Assert 2 — nouvelle allocation et strides canoniques de la shape (3, 2)
    assert materialized.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    assert materialized.shape == torch.Size([3, 2])
    assert materialized.stride() == (2, 1)

    # Assert 3 — mêmes valeurs logiques, nouvel ordre physique
    torch.testing.assert_close(materialized, transposed, rtol=0.0, atol=0.0)
    assert materialized.flatten().tolist() == [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]

    # Assert 4 — sur un tenseur déjà contigu, aucune copie n'est faite
    assert ensure_contiguous(x).untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
