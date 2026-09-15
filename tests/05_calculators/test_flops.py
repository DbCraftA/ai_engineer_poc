"""Section 5.5 / 5.6 — compter les FLOPs : couche linéaire puis étape de decode.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/05_calculators/test_flops.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(48 FLOPs pour un produit 2x3 par 3x4, 1 605 632 pour une projection 896 -> 896,
1 000 000 000 pour une étape de decode) : c'est volontaire. Un test doit énoncer la vérité
attendue, pas la recalculer avec la même formule que le code testé.

Toute la comptabilité de calcul du dépôt tient dans une seule formule, celle du produit
matriciel :

    FLOPs(M, N, K) = 2 x M x N x K

Le facteur 2 vient de ce qu'un élément de sortie s'obtient par une suite de
multiplications-additions : chaque terme du produit scalaire coûte une multiplication PUIS
une addition, soit deux opérations flottantes. Les sections 5.8 à 5.10 diviseront ces FLOPs
par les octets déplacés en 5.7 pour décider quelle ressource limite le moteur.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest


@pytest.mark.tdd
def test_linear_layer_flops_are_estimated_from_matrix_dimensions():
    """Roadmap 5.5 — le coût d'une couche linéaire est 2 x M x N x K, jamais autre chose.

    Objectif d'apprentissage
    ------------------------
    Une couche linéaire, c'est un produit matriciel : une entrée (M, K) rencontre un poids
    (K, N) et produit une sortie (M, N). Chacun des M x N éléments de sortie est un produit
    scalaire de longueur K, donc K multiplications et K additions :

        FLOPs = M x N x K multiplications + M x N x K additions = 2 x M x N x K

    Savoir compter cela à la main, c'est pouvoir prédire le temps de calcul d'un modèle
    avant de l'exécuter, et surtout comparer ce coût aux octets déplacés (5.7) : c'est ce
    rapport, pas le nombre de FLOPs seul, qui dit si le GPU travaille ou attend la mémoire.
    Retiens la dissymétrie fondamentale : le poids ne dépend pas de M, mais le calcul si.
    Un prefill de 128 tokens fait 128 fois plus de FLOPs qu'une étape de decode sur la MÊME
    matrice — même volume de poids lu, 128 fois plus de travail utile.

    Schéma mental
    -------------
        cas jouet :   (M=2, K=3) @ (K=3, N=4) -> (2, 4)
                      2 x 2 x 4 x 3 = 48 FLOPs   (24 multiplications + 24 additions)

        q_proj de Qwen2.5-0.5B, hidden = 896, un seul token en decode :
                      (M=1, K=896) @ (896, 896) -> (1, 896)
                      2 x 1 x 896 x 896 = 1 605 632 FLOPs

        même matrice, prefill d'un prompt de 128 tokens :
                      2 x 128 x 896 x 896 = 205 520 896 FLOPs   (x128)

    Ce que ce test vérifie
    ----------------------
    1. le cas jouet vérifiable de tête : 48 FLOPs pour (2, 3) @ (3, 4) ;
    2. le facteur 2 est bien un facteur 2 : le résultat vaut deux fois le nombre de
       multiplications-additions, soit 2 x 24 ;
    3. la projection q_proj de Qwen2.5-0.5B sur un token vaut 1 605 632 FLOPs, et la
       projection `gate_proj` du MLP (896 -> 4864) en vaut 8 716 288 ;
    4. la linéarité en M : le même poids sur 128 tokens coûte exactement 128 fois plus.

    API à faire émerger (cible roadmap : `src/inference_lab/calculators/flops.py`)
    -----------------------------------------------------------------------------
        def linear_flops(num_tokens: int, in_features: int, out_features: int) -> int: ...

        Convention d'appel : `num_tokens` est le M de la formule (lignes d'entrée),
        `in_features` le K (dimension contractée), `out_features` le N.

    Indice : trois multiplications et un facteur 2, rien d'autre. Pièges : ne compte pas le
    biais (M x N additions, négligeable et absent des projections de Qwen), n'oublie pas le
    facteur 2 (erreur d'un facteur deux sur tout le reste de la section), et ne confonds pas
    K et N — l'expression est symétrique en M, N, K, donc une inversion ne se voit PAS sur
    une matrice carrée comme q_proj, seulement sur une matrice rectangulaire comme gate_proj.
    """

    pytest.skip("Roadmap TDD 5.5 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.calculators.flops import linear_flops

    # Arrange — aucun tenseur, seulement des dimensions entières. Le cas jouet :
    #           `toy_tokens` = 2, `toy_in` = 3, `toy_out` = 4. Puis la configuration
    #           Qwen2.5-0.5B : `hidden` = 896, `intermediate` = 4864, avec deux régimes,
    #           `decode_tokens` = 1 (une étape de génération) et `prefill_tokens` = 128
    #           (un prompt traité d'un coup).

    # Act — demander à l'API les FLOPs du cas jouet, de la projection q_proj
    #       (`hidden` -> `hidden`) en decode puis en prefill, et de la projection gate_proj
    #       (`hidden` -> `intermediate`) en decode. Stocker les résultats dans `toy_flops`,
    #       `q_proj_decode_flops`, `q_proj_prefill_flops` et `gate_proj_decode_flops`.

    # Assert 1 — le cas jouet, vérifiable de tête
    assert toy_flops == 48

    # Assert 2 — le facteur 2 = une multiplication + une addition par terme du produit scalaire
    assert toy_flops == 2 * (toy_tokens * toy_out * toy_in)
    assert toy_flops == 2 * 24

    # Assert 3 — les deux projections réelles de Qwen2.5-0.5B, sur un token
    assert q_proj_decode_flops == 1_605_632
    assert gate_proj_decode_flops == 8_716_288

    # Assert 4 — linéarité en nombre de tokens : le poids est le même, le calcul non
    assert q_proj_prefill_flops == 205_520_896
    assert q_proj_prefill_flops == 128 * q_proj_decode_flops
    assert linear_flops(1, hidden, hidden) == q_proj_decode_flops


@pytest.mark.tdd
def test_model_decode_flops_can_be_estimated_from_architecture():
    """Roadmap 5.6 — une étape de decode coûte environ 2 FLOPs par paramètre.

    Objectif d'apprentissage
    ------------------------
    Additionner les 2 x M x N x K de toutes les couches donne un résultat remarquablement
    simple quand M = 1 (un seul token, le régime du decode) : chaque poids est lu une fois
    et sert exactement à une multiplication-addition. Le total ne dépend donc plus de
    l'architecture, seulement du nombre de paramètres :

        FLOPs_par_token_généré ≈ 2 x nombre_de_paramètres

    C'est l'estimation de coin de table que tout ingénieur inférence connaît par cœur : un
    modèle de 0,5 milliard de paramètres demande ~1 GFLOP par token généré. Mise en regard
    des 1 000 000 000 octets relus à chaque étape (5.7), elle donne l'intensité arithmétique
    de ~1 FLOP/octet de la section 5.8, et donc le diagnostic « bandwidth-bound » de 5.10.

    L'approximation néglige tout ce qui n'est pas un produit poids-activation : softmax,
    RMSNorm, RoPE, et le produit d'attention QK^T/PV dont le coût croît avec la longueur du
    contexte au lieu d'être fixe. À contexte court, ces termes restent sous les 10 %.

    Schéma mental
    -------------
        Qwen2.5-0.5B, arrondi à 500 000 000 paramètres

            1 token généré    : 2 x 500e6 =   1 000 000 000 FLOPs =   1 GFLOP
          100 tokens générés  : 100 x 1e9 = 100 000 000 000 FLOPs = 100 GFLOPs

        cohérence avec 5.5 sur une matrice unique 896 x 4864 = 4 358 144 paramètres :
            2 x 4 358 144 = 8 716 288 = linear_flops(1, 896, 4864)   -> même nombre

    Ce que ce test vérifie
    ----------------------
    1. une étape de decode sur 500 000 000 paramètres vaut 1 000 000 000 FLOPs ;
    2. la cohérence avec la formule de 5.5 : sur un modèle réduit à une seule matrice
       896 x 4864, « 2 x paramètres » et « 2 x M x N x K avec M = 1 » donnent le même
       8 716 288 — les deux comptages ne sont pas deux formules, c'est la même ;
    3. la linéarité en tokens générés : 100 tokens coûtent 100 000 000 000 FLOPs, et le
       coût par token reste CONSTANT (contrairement à la mémoire du cache, section 5.3).

    API à faire émerger (cible roadmap : `src/inference_lab/calculators/flops.py`)
    -----------------------------------------------------------------------------
        def decode_flops_per_token(num_parameters: int) -> int: ...
        def decode_flops(num_parameters: int, num_generated_tokens: int) -> int: ...

        La roadmap indique seulement « calculator » pour ce test : on propose de compléter le
        module concret de 5.5 plutôt que de créer un nouveau fichier.

    Indice : ces deux fonctions sont volontairement triviales — leur intérêt est de NOMMER
    l'approximation et de la rendre citable dans les tests suivants. Piège : ne réintroduis
    pas un facteur `num_layers`, il est déjà contenu dans le nombre de paramètres ; et ne
    confonds pas ce coût avec celui du prefill, qui vaut `2 x paramètres x nombre_de_tokens`
    du prompt.
    """

    pytest.skip("Roadmap TDD 5.6 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.calculators.flops import decode_flops, decode_flops_per_token, linear_flops

    # Arrange — deux comptages de paramètres, en entiers : `num_parameters` = 500 000 000, le
    #           total de Qwen2.5-0.5B arrondi, et `single_matrix_parameters`, le nombre de
    #           poids d'une unique matrice `hidden` x `intermediate` = 896 x 4864, qui servira
    #           de modèle jouet pour recouper 5.5 et 5.6. Prévoir aussi `generated_tokens` = 100.

    # Act — demander à l'API les FLOPs d'une étape de decode pour `num_parameters`, puis ceux
    #       de la génération complète de `generated_tokens` tokens. Stocker les résultats dans
    #       `flops_one_token` et `flops_hundred_tokens`.

    # Assert 1 — 2 FLOPs par paramètre, soit 1 GFLOP par token pour un modèle de 0,5 milliard
    assert flops_one_token == 1_000_000_000
    assert flops_one_token == 2 * num_parameters

    # Assert 2 — même nombre par les deux chemins de calcul, sur un modèle d'une seule matrice
    assert single_matrix_parameters == 4_358_144
    assert decode_flops_per_token(single_matrix_parameters) == 8_716_288
    assert decode_flops_per_token(single_matrix_parameters) == linear_flops(1, 896, 4864)

    # Assert 3 — coût par token constant, coût total linéaire en tokens générés
    assert flops_hundred_tokens == 100_000_000_000
    assert flops_hundred_tokens == 100 * flops_one_token
    assert decode_flops(num_parameters, 1) == flops_one_token
