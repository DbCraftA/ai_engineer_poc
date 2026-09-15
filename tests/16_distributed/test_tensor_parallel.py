"""Section 16.1 / 16.2 — parallélisme de tenseurs : découper un poids, recoller le résultat.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/16_distributed/test_tensor_parallel.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
((8, 8) par shard, (1, 4, 16) après concaténation, tolérance fp32 rtol=1e-5 / atol=1e-6) :
c'est volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même
formule que le code testé.

Dimensions jouets constantes de toute la section 16 : hidden=8, intermediate=16, num_heads=4,
head_dim=2, seq=4, batch=1, tensor_parallel_size=2.

Pourquoi ces tests tournent sur UN SEUL processus
-------------------------------------------------
Ces tests portent le marker `distributed` mais n'appellent JAMAIS
`torch.distributed.init_process_group`, n'utilisent ni NCCL ni plusieurs GPU. On SIMULE le
parallélisme de tenseurs : on découpe les poids en shards (un par « device » simulé), on
calcule chaque contribution locale, puis on recombine à la main. Ce choix isole exactement ce
qui s'apprend ici — QUEL découpage, QUELLE collective — de la plomberie de communication, qui
n'a aucun intérêt pédagogique et rend les tests lents, flaky et impossibles à exécuter sur une
machine à un seul GPU. Le jour où l'on branche NCCL, la mathématique est déjà prouvée : il ne
reste que le transport.

Les deux tests de ce fichier posent l'opposition fondatrice du chapitre : découper un poids sur
la dimension de SORTIE se recombine par CONCATÉNATION (AllGather), découper sur la dimension
d'ENTRÉE se recombine par SOMME (AllReduce). Tout le reste de la section en découle.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
@pytest.mark.distributed
def test_column_parallel_linear_reconstructs_reference_output():
    """Roadmap 16.1 — découper la dimension de SORTIE : chaque device produit des colonnes.

    Objectif d'apprentissage
    ------------------------
    Un modèle qui ne tient pas dans un GPU doit voir ses matrices coupées en morceaux. La
    première façon de couper une couche linéaire `y = x @ W.T` est de couper `W` selon sa
    dimension de SORTIE : le device 0 détient les premières lignes de `W` (donc les premières
    colonnes de `y`), le device 1 les suivantes. Chaque device voit l'entrée `x` en ENTIER et
    produit une TRANCHE de la sortie. Aucune communication n'est nécessaire pendant le calcul :
    il suffit de concaténer les tranches à la fin, et seulement si quelqu'un a besoin de la
    sortie complète. C'est le schéma appliqué à `q_proj`, `k_proj`, `v_proj`, `gate_proj` et
    `up_proj` d'un bloc Transformer : chaque device travaille sur ses propres têtes
    d'attention et sa propre tranche d'intermédiaire, sans jamais parler aux autres.

    Schéma mental
    -------------
        W (intermediate=16, hidden=8)   --split(dim=0)-->   W0 (8, 8)   W1 (8, 8)

        x (batch=1, seq=4, hidden=8) --W0--> y0 (1, 4, 8)   colonnes 0..7  de la sortie
        x (batch=1, seq=4, hidden=8) --W1--> y1 (1, 4, 8)   colonnes 8..15 de la sortie

        cat([y0, y1], dim=-1) == y_ref (1, 4, 16)      <- CONCATÉNATION, pas somme
        chaque device lit x en entier ; aucune collective avant la fin

    Ce que ce test vérifie
    ----------------------
    1. le découpage porte sur la dimension de SORTIE : 2 shards de poids (8, 8) à partir d'un
       poids (16, 8) ;
    2. chaque contribution locale a la shape complète en `seq` mais partielle en sortie
       (1, 4, 8), la concaténation sur la dernière dimension rend (1, 4, 16), et l'ordre des
       rangs fait partie du contrat : la première tranche est la contribution du device 0 ;
    3. cette concaténation reconstitue la référence non parallèle à la tolérance fp32
       rtol=1e-5 / atol=1e-6 ;
    4. un nombre de devices qui ne divise pas la dimension de sortie est refusé.

    À noter sur l'égalité exacte : chaque élément de `y` est le même produit scalaire sur les
    8 mêmes entrées, dans le même ordre — on pourrait donc espérer une égalité bit à bit.
    Mesuré ici, ce n'est PAS le cas (écart max observé ~2e-6 sur 200 tirages) : BLAS choisit un
    découpage en blocs différent selon la taille de la matrice, donc l'ordre réel des additions
    change avec la shape du shard. D'où la tolérance, jamais `torch.equal`, sur toute
    recombinaison numérique de cette section.

    API à faire émerger (cible roadmap : « future distributed », cible proposée
    `src/inference_lab/distributed/tensor_parallel.py`)
    ----------------------------------------------------------------------------
        def shard_column_parallel(
            weight: torch.Tensor, tensor_parallel_size: int
        ) -> list[torch.Tensor]: ...

        def column_parallel_linear(
            x: torch.Tensor, weight_shards: list[torch.Tensor]
        ) -> torch.Tensor: ...

    Indice : convention `torch.nn.Linear`, le poids est de shape (out_features, in_features).
    « Column parallel » nomme les colonnes de la SORTIE, donc le découpage se fait sur la
    dimension 0 du poids : `weight.split(out_features // tensor_parallel_size, dim=0)`. La
    référence non parallèle est `torch.nn.functional.linear(x, weight)`. Lève une `ValueError`
    si `out_features % tensor_parallel_size != 0`. Pas de biais ici : il se découperait comme
    la sortie, exactement comme le poids.
    """

    pytest.skip("Roadmap TDD 16.1 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.distributed.tensor_parallel import (
        column_parallel_linear,
        shard_column_parallel,
    )

    # Arrange — dimensions jouets de la section : hidden=8, intermediate=16, seq=4, batch=1,
    #           tensor_parallel_size=2, tout en `torch.float32` sur CPU.
    #           `x` : tenseur (1, 4, 8) déterministe (seed fixée), vu en ENTIER par chaque
    #           device simulé.
    #           `weight` : poids d'une couche linéaire sans biais, shape (16, 8) au format
    #           `torch.nn.Linear` (out_features, in_features), déterministe.
    #           `reference` : la sortie non parallèle `torch.nn.functional.linear(x, weight)`,
    #           de shape (1, 4, 16) — c'est l'oracle du test.
    #           `weight_shards` : le résultat de `shard_column_parallel(weight, 2)`.

    # Act — calculer `local_outputs`, la liste des sorties locales obtenues en appliquant
    #       `torch.nn.functional.linear(x, shard)` pour chaque shard, puis
    #       `gathered = column_parallel_linear(x, weight_shards)`.

    # Assert 1 — le découpage porte sur la dimension de SORTIE
    assert len(weight_shards) == 2
    assert weight_shards[0].shape == (8, 8)
    assert weight_shards[1].shape == (8, 8)
    assert weight.shape == (16, 8)

    # Assert 2 — chaque device produit une tranche de colonnes, dans l'ordre des rangs
    assert local_outputs[0].shape == (1, 4, 8)
    assert local_outputs[1].shape == (1, 4, 8)
    assert gathered.shape == (1, 4, 16)
    torch.testing.assert_close(gathered[..., 0:8], local_outputs[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gathered[..., 8:16], local_outputs[1], rtol=1e-5, atol=1e-6)

    # Assert 3 — la recombinaison reconstitue la référence mono-device
    assert reference.shape == (1, 4, 16)
    torch.testing.assert_close(gathered, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        column_parallel_linear(x, weight_shards), reference, rtol=1e-5, atol=1e-6
    )

    # Assert 4 — 16 n'est pas divisible par 3 : configuration refusée
    with pytest.raises(ValueError):
        shard_column_parallel(weight, 3)


@pytest.mark.tdd
@pytest.mark.distributed
def test_row_parallel_linear_reconstructs_reference_output():
    """Roadmap 16.2 — découper la dimension d'ENTRÉE : chaque device produit un résultat partiel.

    Objectif d'apprentissage
    ------------------------
    C'est LE point qui bloque tout le monde, alors autant l'énoncer brutalement : ici la
    recombinaison n'est PAS une concaténation, c'est une SOMME. Quand on coupe `W` selon sa
    dimension d'ENTRÉE, le device 0 ne voit que les 8 premières composantes de `x` et le
    device 1 les 8 suivantes. Chacun calcule un produit scalaire INCOMPLET : la shape de sa
    sortie est déjà la shape finale (1, 4, 8), mais la valeur est fausse, car il manque la
    moitié des termes de la somme. Additionner les contributions termine le produit scalaire :
    c'est exactement ce que fait un AllReduce. Retiens la table :

        column parallel (dim de SORTIE coupée)  -> shapes partielles -> CONCATÉNATION
        row parallel    (dim d'ENTRÉE coupée)   -> shape complète    -> SOMME (AllReduce)

    Un raccourci mnémotechnique : si la shape locale est déjà la bonne, c'est qu'il manque des
    termes, donc il faut sommer. Si la shape locale est trop petite, il manque des colonnes,
    donc il faut concaténer. C'est le schéma de `o_proj` et de `down_proj` dans un bloc
    Transformer, dont l'entrée est justement déjà découpée par la couche column parallel qui
    précède : aucune redistribution n'est nécessaire entre les deux.

    Schéma mental
    -------------
        W (hidden=8, intermediate=16)  --split(dim=1)-->  W0 (8, 8)   W1 (8, 8)
        x (1, 4, 16)                   --split(dim=-1)->  x0 (1, 4, 8)   x1 (1, 4, 8)

        x0 --W0--> p0 (1, 4, 8)   partiel : 8 termes sur 16
        x1 --W1--> p1 (1, 4, 8)   partiel : les 8 autres

        p0 + p1 == y_ref (1, 4, 8)     <- SOMME, et p0 seul est FAUX
        cat([p0, p1], dim=-1) -> (1, 4, 16) : la mauvaise shape, le piège classique

    Ce que ce test vérifie
    ----------------------
    1. le découpage porte sur la dimension d'ENTRÉE : 2 shards (8, 8) à partir d'un poids
       (8, 16), et l'entrée est découpée de la même façon ;
    2. chaque contribution locale a DÉJÀ la shape finale (1, 4, 8) — contraste direct avec
       16.1 — mais ne vaut pas la référence : elle est incomplète ;
    3. la SOMME des contributions reconstitue la référence à la tolérance fp32
       rtol=1e-5 / atol=1e-6, et non leur concaténation, qui n'a même pas la bonne shape ;
    4. un nombre de devices qui ne divise pas la dimension d'entrée est refusé.

    Sur la tolérance : la référence somme les 16 produits en une seule réduction, le chemin
    parallèle somme 8 termes, puis 8 autres, puis les deux résultats. L'addition flottante
    n'est pas associative, donc le résultat diffère du dernier bit (écart max observé ~3e-6
    sur 200 tirages). C'est du bruit d'arrondi, pas un bug : on compare avec
    `torch.testing.assert_close`, jamais avec `torch.equal`.

    API à faire émerger (cible roadmap : « future distributed », cible proposée
    `src/inference_lab/distributed/tensor_parallel.py`)
    ----------------------------------------------------------------------------
        def shard_row_parallel(
            weight: torch.Tensor, tensor_parallel_size: int
        ) -> list[torch.Tensor]: ...

        def row_parallel_linear(
            x: torch.Tensor, weight_shards: list[torch.Tensor]
        ) -> torch.Tensor: ...

    Indice : `weight.split(in_features // tensor_parallel_size, dim=1)`, et `row_parallel_linear`
    découpe `x` sur sa dernière dimension avec les mêmes tailles avant de sommer les résultats
    locaux. Lève une `ValueError` si `in_features % tensor_parallel_size != 0`. Deux pièges à
    connaître même s'ils ne sont pas assertés ici : un biais éventuel s'ajoute UNE SEULE FOIS,
    après la somme (sinon il est compté deux fois), et une non-linéarité ne peut pas être
    appliquée avant la somme, car `f(a) + f(b) != f(a + b)`.
    """

    pytest.skip("Roadmap TDD 16.2 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.distributed.tensor_parallel import (
        row_parallel_linear,
        shard_row_parallel,
    )

    # Arrange — dimensions jouets de la section : hidden=8, intermediate=16, seq=4, batch=1,
    #           tensor_parallel_size=2, tout en `torch.float32` sur CPU.
    #           `x` : tenseur (1, 4, 16) déterministe (seed fixée) — l'entrée d'une couche
    #           `down_proj`, c'est-à-dire la sortie d'une couche column parallel.
    #           `weight` : poids sans biais de shape (8, 16) au format `torch.nn.Linear`.
    #           `reference` : la sortie non parallèle `torch.nn.functional.linear(x, weight)`,
    #           de shape (1, 4, 8).
    #           `weight_shards` : le résultat de `shard_row_parallel(weight, 2)`.

    # Act — calculer `partial_outputs`, la liste des contributions locales obtenues en
    #       appliquant `torch.nn.functional.linear(x[..., 0:8], weight_shards[0])` puis
    #       `torch.nn.functional.linear(x[..., 8:16], weight_shards[1])`, et
    #       `reduced = row_parallel_linear(x, weight_shards)`.

    # Assert 1 — le découpage porte sur la dimension d'ENTRÉE
    assert len(weight_shards) == 2
    assert weight_shards[0].shape == (8, 8)
    assert weight.shape == (8, 16)

    # Assert 2 — shape finale dès le calcul local, mais valeur incomplète
    assert partial_outputs[0].shape == (1, 4, 8)
    assert partial_outputs[1].shape == (1, 4, 8)
    assert reduced.shape == (1, 4, 8)
    assert not torch.allclose(partial_outputs[0], reference, rtol=1e-3, atol=1e-3)

    # Assert 3 — c'est la SOMME qui reconstitue la référence, pas la concaténation
    torch.testing.assert_close(reduced, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        row_parallel_linear(x, weight_shards), reference, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        partial_outputs[0] + partial_outputs[1], reference, rtol=1e-5, atol=1e-6
    )
    assert torch.cat(partial_outputs, dim=-1).shape == (1, 4, 16)

    # Assert 4 — 16 n'est pas divisible par 3 : configuration refusée
    with pytest.raises(ValueError):
        shard_row_parallel(weight, 3)
