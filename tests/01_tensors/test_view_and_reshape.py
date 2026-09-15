"""Section 1.7 — `view` contre `reshape` : quand PyTorch a le droit de ne rien copier.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/01_tensors/test_view_and_reshape.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
((4, 8), 32 éléments, 128 octets) : c'est volontaire. Un test doit énoncer la vérité
attendue, pas la recalculer avec la même formule que le code testé.

Cette section est la conclusion pratique des sections 1.4 à 1.6 : `view` exige un layout
compatible et échoue bruyamment sinon, `reshape` accepte tout mais peut copier en silence.
Dans un moteur d'inférence, cette différence décide si un changement de shape est gratuit
ou s'il consomme de la bande passante à chaque token décodé.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_view_does_not_copy_when_layout_allows_it():
    """Roadmap 1.7 — `view` est gratuit, mais seulement si les strides le permettent.

    Objectif d'apprentissage
    ------------------------
    `view` est l'outil de base pour découper et regrouper des axes sans payer une copie :
    fusionner `[batch, seq]` en `[batch * seq]` avant une projection linéaire, ou éclater
    `[batch, seq, hidden]` en `[batch, seq, heads, head_dim]` à l'entrée d'une attention
    multi-têtes. Sa règle est stricte : il refuse de mentir. Si le layout courant ne permet
    pas de réinterpréter les octets sans les déplacer, il lève une `RuntimeError` au lieu de
    copier dans ton dos. C'est une bonne nouvelle : l'échec est un signal de performance,
    il t'indique exactement où un `contiguous()` explicite est nécessaire.

    Schéma mental
    -------------
        hidden       shape (1, 4, 8), strides (32, 8, 1), contigu   128 octets float32
        hidden.view(4, 8)   -> OK, même storage, 0 octet copié

        transposed = hidden.transpose(1, 2)
                     shape (1, 8, 4), strides (32, 1, 8)  -> NON contigu
        transposed.view(32) -> RuntimeError : les 32 valeurs ne se lisent pas d'affilée

    Ce que ce test vérifie
    ----------------------
    1. l'état de départ : `hidden` est contigu, `transposed` ne l'est pas ;
    2. sur le tenseur contigu, la vue `(4, 8)` est autorisée et donne 32 éléments ;
    3. elle ne copie rien : même storage, 128 octets de part et d'autre ;
    4. sur le tenseur transposé, la vue est refusée — prédite `False` par l'API et
       sanctionnée par une `RuntimeError` côté PyTorch ; une shape au nombre d'éléments
       incompatible est refusée de la même façon.

    API à faire émerger (cible roadmap : `src/inference_lab/tensors/layout.py`)
    -------------------------------------------------------------------------
        def can_view(tensor: torch.Tensor, shape: tuple[int, ...]) -> bool: ...
        def as_view(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor: ...

    `can_view` répond « la vue est-elle garantie possible ? » : tenseur contigu ET nombre
    d'éléments identique. C'est une condition suffisante, volontairement simple ; PyTorch
    sait faire un peu mieux sur certains layouts non contigus, ce n'est pas le sujet ici.
    `as_view` délègue à `tensor.view(shape)` et laisse remonter la `RuntimeError`.

    Indice : `tensor.is_contiguous()` et `tensor.numel()` suffisent pour `can_view`. Piège :
    n'attrape PAS la `RuntimeError` dans `as_view` pour la transformer en `None` — le test
    l'attend avec `pytest.raises`, et un échec explicite vaut mieux qu'un retour silencieux.
    """

    pytest.skip("Roadmap TDD 1.7 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.tensors.layout import as_view, can_view

    # Arrange — créer `hidden`, un tenseur contigu de hidden states de shape (1, 4, 8) en
    #           `torch.float32`, puis `transposed`, sa vue obtenue en échangeant les axes
    #           `seq` et `hidden` (axes 1 et 2), qui n'est plus contiguë.

    # Act — construire `grouped`, la vue de `hidden` en shape (4, 8) obtenue via l'API
    #       (les axes `batch` et `seq` sont fusionnés).

    # Assert 1 — l'état de départ des deux layouts
    assert hidden.is_contiguous() is True
    assert transposed.is_contiguous() is False

    # Assert 2 — la vue autorisée : shape demandée, nombre d'éléments conservé
    assert can_view(hidden, (4, 8)) is True
    assert grouped.shape == torch.Size([4, 8])
    assert grouped.numel() == 32

    # Assert 3 — aucun octet copié : un seul storage pour les deux lectures
    assert grouped.untyped_storage().data_ptr() == hidden.untyped_storage().data_ptr()
    assert grouped.untyped_storage().nbytes() == 128
    assert hidden.untyped_storage().nbytes() == 128

    # Assert 4 — les deux refus : layout incompatible, puis nombre d'éléments incompatible
    assert can_view(transposed, (32,)) is False
    with pytest.raises(RuntimeError):
        as_view(transposed, (32,))
    assert can_view(hidden, (5, 7)) is False


@pytest.mark.tdd
def test_reshape_may_materialize_when_required():
    """Roadmap 1.7 — `reshape` réussit toujours, au prix d'une copie parfois invisible.

    Objectif d'apprentissage
    ------------------------
    `reshape` est le compromis pragmatique : vue gratuite si le layout s'y prête, copie
    sinon. Pratique pour écrire du code qui marche, dangereux pour écrire du code rapide,
    parce que le coût n'apparaît nulle part dans la ligne appelante. Dans une boucle de
    décodage exécutée une fois par token, un `reshape` qui matérialise silencieusement une
    activation transposée est exactement le genre de coût qu'un profilage révèle trop tard.
    La règle de travail du projet : `view` quand tu veux garantir zéro copie et échouer
    sinon, `reshape` quand tu acceptes la copie en connaissance de cause.

    Schéma mental
    -------------
        hidden = arange(32) vu en (1, 4, 8) float32, contigu
                 valeur en (0, t, c) = t * 8 + c

        reshape(hidden, (4, 8))     -> vue gratuite, storage partagé

        transposed = hidden.transpose(1, 2)   shape (1, 8, 4), non contigu
        reshape(transposed, (32,))  -> copie : nouveau storage de 128 octets
                 lecture logique par colonnes : [0, 8, 16, 24, 1, 9, 17, 25, ...]

    Ce que ce test vérifie
    ----------------------
    1. sur un tenseur contigu, `reshape` ne copie pas : le storage reste partagé ;
    2. là où `view` échouait (section précédente), `reshape` réussit et rend bien
       32 éléments en shape `(32,)`, ou `(8, 4)` si l'on abandonne l'axe `batch` ;
    3. il a fallu matérialiser : le résultat ne partage plus le storage de la source ;
    4. les valeurs sont celles de la lecture LOGIQUE du tenseur transposé : les 8
       premières sont `[0, 8, 16, 24, 1, 9, 17, 25]`, pas `[0, 1, 2, 3, 4, 5, 6, 7]`.

    API à faire émerger (cible roadmap : `src/inference_lab/tensors/layout.py`)
    -------------------------------------------------------------------------
        def as_reshaped(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor: ...

    Le prédicat `shares_storage` est celui déjà écrit en section 1.3
    (`src/inference_lab/tensors/inspection.py`) : on le réutilise ici comme instrument de
    mesure, c'est le premier module de la roadmap qui en resservira un autre.

    Indice : `tensor.reshape(shape)` fait le travail ; ce qui compte est de savoir observer
    APRÈS coup s'il y a eu copie, en comparant `untyped_storage().data_ptr()`. Piège :
    ne conclus jamais à l'absence de copie sur la seule base des valeurs, qui sont
    identiques dans les deux cas — seule l'adresse du storage tranche.
    """

    pytest.skip("Roadmap TDD 1.7 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.tensors.inspection import shares_storage
    from inference_lab.tensors.layout import as_reshaped

    # Arrange — créer `hidden`, un tenseur contigu de shape (1, 4, 8) en `torch.float32` dont
    #           les 32 valeurs sont les entiers croissants à partir de 0, puis `transposed`,
    #           sa vue non contiguë obtenue en échangeant les axes 1 et 2.

    # Act — reshaper `hidden` en (4, 8) dans `grouped`, puis `transposed` en (32,) dans
    #       `materialized`, via l'API.

    # Assert 1 — sur un tenseur contigu, reshape est une simple vue
    assert grouped.shape == torch.Size([4, 8])
    assert shares_storage(hidden, grouped) is True

    # Assert 2 — là où `view` lève une RuntimeError, `reshape` répond
    assert materialized.shape == torch.Size([32])
    assert materialized.numel() == 32
    assert as_reshaped(transposed, (8, 4)).shape == torch.Size([8, 4])

    # Assert 3 — la réponse a coûté une copie : le storage n'est plus partagé
    assert shares_storage(transposed, materialized) is False
    assert materialized.untyped_storage().nbytes() == 128

    # Assert 4 — l'ordre est celui de la lecture logique du tenseur transposé
    assert materialized[:8].tolist() == [0.0, 8.0, 16.0, 24.0, 1.0, 9.0, 17.0, 25.0]
