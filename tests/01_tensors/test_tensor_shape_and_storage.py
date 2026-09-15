"""Section 1.2 / 1.3 — shape logique, nombre d'éléments et storage partagé.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/01_tensors/test_tensor_shape_and_storage.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
((1, 4, 8), 32 éléments, 48 octets) : c'est volontaire. Un test doit énoncer la vérité
attendue, pas la recalculer avec la même formule que le code testé.

Cette section installe la distinction fondamentale entre la vue logique d'un tenseur
(sa `shape`, ce que le modèle croit manipuler) et sa mémoire réelle (son *storage*, un
bloc d'octets plat). Toute la roadmap repose dessus : un KV cache est un gros storage
alloué une fois, que l'on relit sous des shapes différentes sans jamais recopier.
La section 1.4 y ajoutera les strides, qui expliquent COMMENT on passe de l'une à l'autre.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_tensor_shape_represents_logical_dimensions():
    """Roadmap 1.2 — la `shape` nomme les axes du calcul, pas les octets.

    Objectif d'apprentissage
    ------------------------
    En inférence LLM, presque tous les tenseurs qui traversent le modèle portent la même
    convention à trois axes : `[batch, sequence, hidden]`, souvent noté `[B, T, C]`. Les
    embeddings sortent en `[B, T, C]`, chaque bloc Transformer entre et sort en `[B, T, C]`,
    le LM head transforme le dernier axe en vocabulaire. Savoir lire une shape, et surtout
    savoir DIRE ce que signifie chaque position, évite les bugs les plus coûteux du projet :
    un `transpose` oublié entre `[B, T, heads, head_dim]` et `[B, heads, T, head_dim]`.

    Schéma mental
    -------------
        hidden states :  (batch=1, seq=4, hidden=8)
                          |        |       |
                          |        |       +-- taille du vecteur par token
                          |        +---------- 4 tokens dans le prompt
                          +------------------- une seule séquence en vol

        len(shape) == 3, shape[0] == 1, shape[1] == 4, shape[2] == 8

    Ce que ce test vérifie
    ----------------------
    1. le tenseur conserve exactement la shape demandée, `(1, 4, 8)`, et son rang est 3 ;
    2. l'API nomme les trois axes et renvoie `{"batch": 1, "seq": 4, "hidden": 8}` ;
    3. les valeurs nommées sont bien celles lues dans `tensor.shape`, qui sert d'oracle ;
    4. un tenseur qui n'a pas trois axes n'est pas des hidden states : l'API refuse
       explicitement avec une `ValueError` plutôt que de renvoyer un résultat douteux.

    API à faire émerger (cible roadmap : `src/inference_lab/tensors/inspection.py`)
    -----------------------------------------------------------------------------
        def hidden_state_dims(tensor: torch.Tensor) -> dict[str, int]: ...

    Indice : `tensor.shape` est un `torch.Size`, c'est-à-dire un tuple d'entiers ;
    `tensor.dim()` (ou `len(tensor.shape)`) donne le rang. Piège : `tensor.size()` sans
    argument renvoie la shape complète, `tensor.size(1)` un seul entier — ne mélange pas
    les deux. La `ValueError` de l'assert 4 fait partie du contrat, pas un bonus.
    """

    pytest.skip("Roadmap TDD 1.2 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.tensors.inspection import hidden_state_dims

    # Arrange — créer `hidden`, un tenseur de hidden states de shape (1, 4, 8) en
    #           `torch.float32` (contenu indifférent : seule la shape est observée), et
    #           `flat`, un tenseur 1D de 8 valeurs qui servira de contre-exemple à 1 seul axe.

    # Act — demander à l'API les dimensions nommées de `hidden` dans `dims`.

    # Assert 1 — la shape logique attendue, écrite en dur
    assert hidden.shape == torch.Size([1, 4, 8])
    assert hidden.dim() == 3

    # Assert 2 — l'API nomme les axes de la convention [B, T, C]
    assert dims == {"batch": 1, "seq": 4, "hidden": 8}

    # Assert 3 — cohérence avec l'oracle PyTorch
    assert dims["batch"] == hidden.shape[0]
    assert dims["seq"] == hidden.shape[1]
    assert dims["hidden"] == hidden.shape[2]

    # Assert 4 — un tenseur de rang 1 n'est pas des hidden states : refus explicite
    with pytest.raises(ValueError):
        hidden_state_dims(flat)


@pytest.mark.tdd
def test_numel_is_product_of_dimensions():
    """Roadmap 1.2 — `numel` est le produit des dimensions, quel que soit le découpage.

    Objectif d'apprentissage
    ------------------------
    `numel` est la quantité qui compte pour la mémoire et la bande passante : combinée au
    `dtype` (section 1.9), elle donne les octets. Deux tenseurs de shapes très différentes
    peuvent contenir exactement le même nombre d'éléments, donc coûter exactement la même
    mémoire : c'est ce qui rend légitime de réinterpréter un KV cache `[B, heads, T, dim]`
    comme un bloc plat, ou de regrouper `[B, T, C]` en `[B * T, C]` avant une projection
    linéaire — une opération faite à chaque couche d'un Transformer.

    Schéma mental
    -------------
        (1, 4, 8)        -> 1 x 4 x 8 = 32 éléments
        (4, 8)           -> 4 x 8     = 32 éléments  (même storage, autre découpage)
        32 éléments float32 -> 32 x 4 = 128 octets

    Ce que ce test vérifie
    ----------------------
    1. le nombre d'éléments attendu, calculé à la main : 32 pour `(1, 4, 8)` ;
    2. l'API est cohérente avec l'oracle PyTorch `tensor.numel()` ;
    3. regrouper les axes en `(4, 8)` ne change pas le nombre d'éléments ;
    4. la conséquence mémoire : 32 éléments en float32 pèsent 128 octets.

    API à faire émerger (cible roadmap : `src/inference_lab/tensors/inspection.py`)
    -----------------------------------------------------------------------------
        def element_count(tensor: torch.Tensor) -> int: ...

    Indice : `tensor.numel()` répond déjà ; l'intérêt du module est de nommer le concept
    et de renvoyer un `int` Python, comparable à un calcul fait à la main. Tu peux aussi
    le vérifier avec `math.prod(tensor.shape)`. Piège : ne renvoie pas un tenseur 0-d,
    `== 32` deviendrait un test sur un tenseur et non sur un entier.
    """

    pytest.skip("Roadmap TDD 1.2 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.tensors.inspection import element_count

    # Arrange — créer `hidden`, un tenseur `torch.float32` de shape (1, 4, 8), puis `grouped`,
    #           le MÊME tenseur réinterprété en (4, 8) avec `view` (batch et seq fusionnés).

    # Act — demander à l'API le nombre d'éléments de `hidden` puis de `grouped`.

    # Assert 1 — la valeur attendue, calculée à la main
    assert element_count(hidden) == 32

    # Assert 2 — cohérence avec l'oracle PyTorch
    assert element_count(hidden) == hidden.numel()
    assert isinstance(element_count(hidden), int)

    # Assert 3 — regrouper les axes ne crée ni ne détruit d'élément
    assert grouped.shape == torch.Size([4, 8])
    assert element_count(grouped) == 32

    # Assert 4 — conséquence mémoire immédiate en float32
    assert element_count(hidden) * hidden.element_size() == 128


@pytest.mark.tdd
def test_view_shares_storage_with_source_tensor():
    """Roadmap 1.3 — une `view` réinterprète un storage existant, elle n'alloue rien.

    Objectif d'apprentissage
    ------------------------
    C'est le mécanisme qui rend l'inférence tenable en mémoire. Un KV cache de plusieurs
    gigaoctets est alloué une fois, puis lu comme `[B, heads, T, head_dim]` par l'attention,
    comme `[B * heads, T, head_dim]` par un kernel batché, sans jamais copier un octet.
    Corollaire à intégrer tout de suite : deux vues du même storage sont deux fenêtres sur
    la même mémoire — écrire dans l'une modifie l'autre. C'est exactement ce qu'on veut
    quand on écrit la clé d'un nouveau token dans le cache, et une source de bugs subtils
    partout ailleurs.

    Schéma mental
    -------------
        flat (12 valeurs float32)  storage : [ 0 1 2 3 4 5 6 7 8 9 10 11 ]  48 octets
                                              ^ un seul bloc, un seul data_ptr
        matrix = flat.view(3, 4)   ->  shape (3, 4), MÊME storage, 0 octet de plus

            matrix[0] = storage[0:4]
            matrix[1] = storage[4:8]
            matrix[2] = storage[8:12]   donc matrix[2, 3] EST flat[11]

    Ce que ce test vérifie
    ----------------------
    1. la vue et sa source partagent le même storage (même adresse de données) ;
    2. écrire dans la vue est visible depuis la source : c'est la même mémoire ;
    3. la vue n'ajoute aucun octet : le storage fait 48 octets des deux côtés ;
    4. les shapes logiques restent pourtant différentes, `(3, 4)` contre `(12,)`.

    API à faire émerger (cible roadmap : `src/inference_lab/tensors/inspection.py`)
    -----------------------------------------------------------------------------
        def storage_data_ptr(tensor: torch.Tensor) -> int: ...
        def shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool: ...

    Indice : compare `tensor.untyped_storage().data_ptr()`, pas `tensor.data_ptr()` (qui
    inclut l'offset de la vue) et pas `tensor.storage()` (API dépréciée). Piège : deux
    tenseurs peuvent avoir des valeurs identiques et des storages distincts — c'est
    l'adresse qui tranche, jamais l'égalité des contenus.
    """

    pytest.skip("Roadmap TDD 1.3 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.tensors.inspection import shares_storage, storage_data_ptr

    # Arrange — créer `flat`, un tenseur 1D contigu de 12 valeurs `torch.float32` construit
    #           avec `torch.arange` (la valeur d'une case est donc égale à son indice), puis
    #           `matrix`, une `view` de `flat` en shape (3, 4).

    # Act — écrire la valeur sentinelle 99.0 dans la dernière case de `matrix`, en ligne 2
    #       colonne 3, et observer ce que devient `flat`.

    # Assert 1 — même storage, donc même adresse de données
    assert shares_storage(flat, matrix) is True
    assert storage_data_ptr(matrix) == storage_data_ptr(flat)

    # Assert 2 — une seule mémoire : l'écriture faite via la vue se lit dans la source
    assert matrix[2, 3].item() == 99.0
    assert flat[11].item() == 99.0

    # Assert 3 — la vue ne coûte aucun octet supplémentaire
    assert matrix.untyped_storage().nbytes() == 48
    assert flat.untyped_storage().nbytes() == 48

    # Assert 4 — deux lectures logiques différentes du même bloc d'octets
    assert flat.shape == torch.Size([12])
    assert matrix.shape == torch.Size([3, 4])


@pytest.mark.tdd
def test_clone_owns_independent_storage():
    """Roadmap 1.3 — `clone` alloue un nouveau storage et coupe le lien avec la source.

    Objectif d'apprentissage
    ------------------------
    C'est l'opération inverse de la `view`, et son prix est réel : `clone` recopie tous les
    octets. Savoir quand on partage et quand on copie est le cœur de l'optimisation mémoire
    d'un moteur d'inférence. On clone quand on a besoin d'un instantané qui ne bougera plus
    (garder les logits d'un pas de décodage, sauvegarder un état avant un rollback de
    speculative decoding) ; on partage partout ailleurs. Cloner par réflexe un KV cache à
    chaque token, c'est transformer un moteur mémoire-bound en moteur inutilisable.

    Schéma mental
    -------------
        source (2, 3) float32, rempli de zéros    storage A : 6 x 4 = 24 octets
        copy = source.clone()                     storage B : 24 octets, adresse ≠ A

        copy[0, 0] = 99.0   ->   copy   = [[99, 0, 0], [0, 0, 0]]
                                 source = [[ 0, 0, 0], [0, 0, 0]]   (intact)

    Ce que ce test vérifie
    ----------------------
    1. le clone possède son propre storage : les adresses de données diffèrent ;
    2. écrire dans le clone ne modifie pas la source, restée à zéro ;
    3. le clone garde shape, dtype et contiguïté de la source ;
    4. la copie a bien un coût : 24 octets alloués en plus, comme la source.

    API à faire émerger (cible roadmap : `src/inference_lab/tensors/inspection.py`)
    -----------------------------------------------------------------------------
        def shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool: ...

    Indice : c'est la même fonction que pour la `view`, exercée sur le cas opposé — un bon
    prédicat doit répondre `True` ET `False`. Piège : `source.detach()`, `source.to("cpu")`
    sur un tenseur déjà CPU ou `source[:]` NE copient pas ; seuls `clone()`, `contiguous()`
    sur un tenseur non contigu ou un changement de dtype allouent un nouveau storage.
    """

    pytest.skip("Roadmap TDD 1.3 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.tensors.inspection import shares_storage

    # Arrange — créer `source`, un tenseur (2, 3) `torch.float32` entièrement nul et contigu,
    #           puis `copy`, un clone indépendant de `source`.

    # Act — écrire la valeur sentinelle 99.0 dans la première case de `copy` (ligne 0,
    #       colonne 0), et regarder si `source` a bougé.

    # Assert 1 — deux storages distincts
    assert shares_storage(source, copy) is False
    assert copy.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()

    # Assert 2 — la source est protégée de l'écriture faite dans le clone
    assert copy[0, 0].item() == 99.0
    assert source[0, 0].item() == 0.0
    assert source.sum().item() == 0.0

    # Assert 3 — le clone reste le même tenseur du point de vue logique
    assert copy.shape == torch.Size([2, 3])
    assert copy.dtype is torch.float32
    assert copy.is_contiguous()

    # Assert 4 — la copie a un coût mémoire, identique à celui de la source
    assert copy.untyped_storage().nbytes() == 24
    assert source.untyped_storage().nbytes() == 24
