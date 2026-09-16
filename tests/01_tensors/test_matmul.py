"""Section 1.10 / 1.11 — matmul : shapes M/N/K, coût en FLOPs et GEMM contre GEMV.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/01_tensors/test_matmul.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
((4, 8), 512 FLOPs, intensité 1.0) : c'est volontaire. Un test doit énoncer la vérité
attendue, pas la recalculer avec la même formule que le code testé.

Cette section installe le vocabulaire de coût de tout le reste du projet. Un Transformer
est essentiellement une pile de multiplications matricielles : les projections Q/K/V, la
sortie d'attention, les deux couches du MLP, le LM head. Deux nombres suffisent à raisonner
sur leur performance : les FLOPs (`2 * M * N * K`) et les octets déplacés. Leur rapport
sépare le prefill (GEMM, compute-bound) du decode (GEMV, memory-bound), la distinction la
plus importante de toute l'inférence LLM.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_matrix_multiplication_produces_expected_shape():
    """Roadmap 1.10 — dans `(M, K) @ (K, N)`, la dimension contractée K disparaît du résultat.

    Objectif d'apprentissage
    ------------------------
    Toutes les couches denses d'un LLM se lisent avec ces trois lettres : `M` compte les
    tokens traités simultanément, `K` la dimension d'entrée, `N` la dimension de sortie.
    Une projection Q de Qwen2.5-0.5B, c'est `(M=nb_tokens, K=896) @ (K=896, N=896)`. Fixer
    cette convention maintenant évite de deviner plus tard qui doit être transposé : la
    seule contrainte est que les deux `K` se rencontrent, et le résultat vaut `(M, N)`.
    C'est aussi la clé de la lecture du prefill (M grand) contre le decode (M = 1).

    Schéma mental
    -------------
        hidden (M=4, K=8)  @  w_q (K=8, N=8)   ->  (M=4, N=8)
                     ^                 ^
                     +--- K commun ----+   K est contracté, il ne survit pas

        cas rectangulaire :  (4, 8) @ (8, 3) -> (4, 3)
        cas du decode     :  (1, 8) @ (8, 8) -> (1, 8)   un seul token en vol
        cas invalide      :  (4, 8) @ (7, 3) -> refusé, 8 != 7

    Ce que ce test vérifie
    ----------------------
    1. le produit réel calculé par PyTorch a la shape `(4, 8)` ;
    2. le calculateur prédit cette shape à partir des seules dimensions, sans allouer,
       y compris pour un cas rectangulaire `(4, 8) @ (8, 3) -> (4, 3)` ;
    3. le cas du decode, `(1, 8) @ (8, 8) -> (1, 8)` : une seule ligne de sortie ;
    4. un `K` incohérent est une erreur : `ValueError` côté calculateur, `RuntimeError`
       côté PyTorch.

    API à faire émerger (cible roadmap : `src/inference_lab/calculators/flops.py`)
    ----------------------------------------------------------------------------
        def matmul_output_shape(lhs: tuple[int, int], rhs: tuple[int, int]) -> tuple[int, int]: ...

    Indice : la fonction tient en une comparaison (`lhs[1] == rhs[0]`) et un tuple de retour.
    Elle vit dans le calculateur de FLOPs parce que c'est elle qui valide le triplet
    (M, N, K) avant tout chiffrage. Piège : `torch.Size([4, 8]) == (4, 8)` est vrai
    (`torch.Size` est un tuple), mais renvoie un vrai `tuple` d'entiers Python, pas un
    `torch.Size`, pour que le calculateur reste utilisable sans tenseur.
    """

    from inference_lab.calculators.flops import matmul_output_shape

    # Arrange — créer `hidden`, un tenseur (4, 8) `torch.float32` représentant 4 tokens de
    #           dimension 8, et `w_q`, une matrice de poids (8, 8) dans le même dtype (contenu
    #           indifférent : seules les shapes comptent ici).

    hidden = torch.randn(4,8,dtype=torch.float32)
    wq = torch.randn(8,8,dtype=torch.float32)

    # Act — calculer le produit matriciel `hidden @ w_q` dans `out`, et demander au
    #       calculateur la shape prédite pour ce même produit, dans `predicted`.

    out = hidden @ wq
    predicted = out.size()

    # Assert 1 — la shape réelle du produit, écrite en dur
    assert out.shape == torch.Size([4, 8])

    # Assert 2 — le calculateur prédit la même chose sans rien allouer
    assert predicted == (4, 8)
    assert matmul_output_shape((4, 8), (8, 3)) == (4, 3)

    # Assert 3 — le cas du decode : un seul token, une seule ligne de sortie
    assert matmul_output_shape((1, 8), (8, 8)) == (1, 8)

    # Assert 4 — les deux dimensions contractées doivent coïncider
    with pytest.raises(ValueError):
        matmul_output_shape((4, 8), (7, 3))
    with pytest.raises(RuntimeError):
        hidden @ hidden


@pytest.mark.tdd
def test_matmul_flops_can_be_estimated_from_mnk():
    """Roadmap 1.10 — chaque élément de sortie coûte K multiplications et K additions.

    Objectif d'apprentissage
    Une opération flottante (FLOP) est une opération arithmétique de base effectuée sur des nombres à virgule flottante (float32, float16, bfloat16, etc.) :
    Une addition : a + b
    Une multiplication : a × b
    Une multiplication-accumulation (FMA) : a × b + c (compte souvent pour 2 FLOPs)
    ⚠️ Attention à la nuance :
        FLOP (singulier) = une opération
        FLOPs (pluriel, minuscule) = plusieurs opérations (une quantité totale)
        FLOPS = FLOP par Seconde (une vitesse, un débit)
    ------------------------
    Le coût arithmétique d'un matmul dense se chiffre sans le lancer :

        FLOPs = 2 * M * N * K

    Le facteur 2 vient du produit scalaire : pour chacun des `M * N` éléments de sortie, on
    fait `K` multiplications et `K` additions. C'est la formule qui permet de comparer une
    mesure de temps à la puissance crête d'un GPU, d'estimer le coût d'un prefill avant de
    l'exécuter, et de vérifier qu'un kernel maison n'est pas 10 fois trop lent. On la
    réutilisera pour chiffrer une couche entière, puis le modèle complet.

    Schéma mental
    -------------
        (M=4, K=8) @ (K=8, N=8)  ->  M*N = 32 éléments de sortie
                                     chacun : 8 mul + 8 add = 16 FLOPs
                                     total  : 32 x 16 = 512 FLOPs

        decode (M=1) : 2 * 1 * 8 * 8 = 128 FLOPs   -> 4 tokens de prefill coûtent 4 x 128
        échelle Qwen2.5-0.5B, une projection sur 1 token, hidden = 896 :
                       2 * 1 * 896 * 896 = 1 605 632 FLOPs

    Ce que ce test vérifie
    ----------------------
    1. la valeur attendue pour `M=4, N=8, K=8`, calculée à la main : 512 FLOPs ;
    2. la décomposition du produit scalaire : un seul élément de sortie coûte `2 * K`,
       soit 16 FLOPs ;
    3. la linéarité en `M` : le decode d'un token coûte 128 FLOPs, et 4 tokens de prefill
       coûtent exactement 4 fois plus ;
    4. l'ordre de grandeur réel d'une projection de Qwen2.5-0.5B : 1 605 632 FLOPs.

    API à faire émerger (cible roadmap : `src/inference_lab/calculators/flops.py`)
    ----------------------------------------------------------------------------
        def matmul_flops(m: int, n: int, k: int) -> int: ...

    Indice : une seule ligne, `2 * m * n * k`, mais renvoie un `int` et respecte l'ordre des
    paramètres — les trois lettres se confondent vite. Piège : ce chiffre est un coût
    ARITHMÉTIQUE théorique ; il ne dit rien du temps réel, qui dépend aussi des octets
    déplacés (c'est l'objet du test suivant).
    """


    from inference_lab.calculators.flops import matmul_flops

    # Arrange — poser les trois dimensions du cas de référence dans `m`, `n` et `k` :
    #           4 tokens, une matrice de poids carrée de dimension 8 (donc n = k = 8).
    m = 4
    n = 8
    k = 8


    # Act — demander au calculateur les FLOPs de ce matmul, dans `flops`.
    flops = matmul_flops(m, n, k)

    # Assert 1 — la valeur attendue, calculée à la main : 2 x 4 x 8 x 8
    assert flops == 512

    # Assert 2 — un seul élément de sortie coûte 2 * K
    assert matmul_flops(1, 1, 8) == 16

    # Assert 3 — linéarité en M : le prefill de 4 tokens vaut 4 decodes
    assert matmul_flops(1, 8, 8) == 128
    assert matmul_flops(4, 8, 8) == 4 * matmul_flops(1, 8, 8)

    # Assert 4 — ordre de grandeur réel : une projection de Qwen2.5-0.5B sur 1 token
    assert matmul_flops(1, 896, 896) == 1_605_632


@pytest.mark.tdd
def test_gemv_is_matmul_with_single_output_row_or_vector_workload():
    """Roadmap 1.11 — à M=1, le matmul devient un GEMV : même mémoire lue, 4 fois moins de calcul.

    Objectif d'apprentissage
    ------------------------
    C'est LA raison pour laquelle générer un token coûte cher alors que le calcul est
    ridicule. Un GEMM (matrice x matrice, `M` grand) lit une matrice de poids une fois et
    l'amortit sur `M` tokens : il sature les unités de calcul, il est *compute-bound*. Un
    GEMV (matrice x vecteur, `M = 1`) lit exactement les mêmes poids pour un seul token :
    le GPU passe son temps à attendre la mémoire, il est *memory-bound*. Le decode
    autorégressif d'un LLM est une longue suite de GEMV, un par token et par couche — d'où
    les techniques de batching et de speculative decoding, qui cherchent toutes à
    reconstituer un `M` supérieur à 1.

    Schéma mental
    -------------
        Pour effectuer un matmul Input (M, K) @ Weights (K, N) = Output (M, N), le GPU doit déplacer trois matrices entre la mémoire (HBM) et les unités de calcul :
        
        1. LIRE   la matrice d'entrée   : taille M × K
        2. LIRE   la matrice de poids   : taille K × N
        3. ÉCRIRE la matrice de sortie  : taille M × N

        intensité arithmétique = FLOPs / octets déplacés
        octets = (M*K + K*N + M*N) * octets_par_élément     (entrée + poids + sortie)

        GEMM  hidden (4, 8) @ weights (8, 8), float32 :
              FLOPs = 512 ; éléments = 32 + 64 + 32 = 128 -> 512 octets ; intensité = 1.0
        GEMV  token  (8,)   @ weights (8, 8), float32 :
              FLOPs = 128 ; éléments =  8 + 64 +  8 =  80 -> 320 octets ; intensité = 0.4

        Mêmes 64 poids lus dans les deux cas : ils pèsent 64/80 = 80 % du trafic du GEMV.
            Trafic total GEMV = 8 + 64 +  8 = 80 elements dont 64 c'est les poids du modele => 64/80 = 80% 

        GEMV : 80% du trafic mémoire = lecture des poids et  20% du trafic mémoire = input + output (les vraies données utiles)

        Le GPU passe l'essentiel de son temps à déplacer des poids depuis la mémoire, alors que la quantité de calcul utile associée (le FLOPs réellement productif) reste minime.
        Alors que pour GEMM c'est 64/128 = 50%

        Puisque les poids représentent une proportion écrasante du trafic mémoire en GEMV (le cas du decode token par token), augmenter M (traiter plusieurs tokens à la fois via le batching) permet de :

            Garder le coût de lecture des poids fixe (toujours 64 éléments)
                    Diluer ce coût fixe sur davantage de calcul utile

        C'est exactement la logique derrière le batching : lire une fois les poids en mémoire, puis les réutiliser pour plusieurs tokens simultanément, plutôt que de les relire à chaque token individuellement.
        

    Ce que ce test vérifie
    ----------------------
    1. les shapes : le GEMM sort en `(4, 8)`, le GEMV sort un vecteur 1D de shape `(8,)` ;
    2. les FLOPs : 512 contre 128, soit un rapport de 4 alors que les poids lus sont
       identiques (64 valeurs dans les deux cas) ;
    3. l'intensité arithmétique en float32 : 1.0 pour le GEMM, 0.4 pour le GEMV ;
    4. la conclusion : l'intensité du GEMV est strictement plus faible, et son trafic est
       dominé par les poids — c'est la signature d'un régime memory-bound.

    API à faire émerger (cible « documentation/calculators », chemin proposé :
    `src/inference_lab/calculators/flops.py`)
    ------------------------------------------------------------------------
        def matmul_arithmetic_intensity(m: int, n: int, k: int, bytes_per_element: int)
            -> float: ...

    Indice : réutilise `matmul_flops` pour le numérateur et compte les octets comme
    `(m * k + k * n + m * n) * bytes_per_element`. En float32, `bytes_per_element` vaut 4 ;
    passer en FP16 (2 octets) divise les octets par deux et double donc l'intensité, ce qui
    explique pourquoi la quantification aide surtout le decode. Piège : `0.4` n'est pas
    représentable exactement en binaire, compare avec `pytest.approx`.
    """

    from inference_lab.calculators.flops import matmul_arithmetic_intensity, matmul_flops

    # Arrange — créer `weights`, une matrice de poids (8, 8) en `torch.float32`, `hidden`, un
    #           bloc de 4 tokens de shape (4, 8) pour le régime prefill, et `token`, un unique
    #           token de shape (8,) — un vecteur 1D, pas une matrice à une ligne.

    weights = torch.randn(8,8,dtype=torch.float32)
    hidden = torch.randn(4,8,dtype=torch.float32)
    token = torch.randn(8,dtype=torch.float32)

    # Act — calculer `gemm_out`, le produit de `hidden` par `weights`, et `gemv_out`, le
    #       produit de `token` par `weights`.
    gemm_out =  hidden @ weights 
    gemv_out = weights @ token

    # Assert 1 — GEMM et GEMV : la sortie perd un axe quand l'entrée est un vecteur
    assert gemm_out.shape == torch.Size([4, 8])
    assert gemv_out.shape == torch.Size([8])
    assert gemv_out.dim() == 1

    # Assert 2 — 4 fois moins de calcul pour exactement les mêmes poids lus
    assert matmul_flops(4, 8, 8) == 512
    assert matmul_flops(1, 8, 8) == 128
    assert weights.numel() == 64

    # Assert 3 — intensité arithmétique en float32 : 4 octets par élément
    assert matmul_arithmetic_intensity(4, 8, 8, 4) == pytest.approx(1.0)
    assert matmul_arithmetic_intensity(1, 8, 8, 4) == pytest.approx(0.4)

    # Assert 4 — le GEMV est memory-bound : moins d'intensité, trafic dominé par les poids
    assert matmul_arithmetic_intensity(1, 8, 8, 4) < matmul_arithmetic_intensity(4, 8, 8, 4)
    assert matmul_arithmetic_intensity(1, 8, 8, 2) == pytest.approx(0.8)
