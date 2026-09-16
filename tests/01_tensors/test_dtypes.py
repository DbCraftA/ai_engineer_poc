"""Section 1.8 / 1.9 — dtype, précision et mémoire d'un tenseur.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/01_tensors/test_dtypes.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(4, 2, 24 octets, ...) : c'est volontaire. Un test doit énoncer la vérité attendue,
pas la recalculer avec la même formule que le code testé.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import math

import pytest
import torch


@pytest.mark.tdd
def test_dtype_controls_bytes_per_element():
    """Roadmap 1.8 — le `dtype` décide du coût mémoire d'UN élément.

    Objectif d'apprentissage
    ------------------------
    Un tenseur ne stocke pas des nombres abstraits, il stocke des octets. Le `dtype`
    fixe le nombre d'octets par élément, donc la mémoire totale du tenseur, donc aussi
    la bande passante nécessaire pour le lire :

        float32  -> 4 octets        float16 -> 2 octets
        bfloat16 -> 2 octets        int8    -> 1 octet

    C'est la brique de base de tous les calculs de mémoire du dépôt (poids, KV cache,
    activations) : passer un modèle de FP32 à FP16 divise sa mémoire par deux sans
    changer une seule shape.

    Schéma mental
    -------------
        torch.empty(2, 3, dtype=torch.float32)  ->  6 éléments x 4 octets = 24 octets
        torch.empty(2, 3, dtype=torch.float16)  ->  6 éléments x 2 octets = 12 octets
        même shape, même sémantique, mémoire divisée par 2

    Ce que ce test vérifie
    ----------------------
    1. le tenseur conserve exactement le `dtype` demandé à la construction ;
    2. `bytes_per_element(dtype)` renvoie 4 pour float32, 2 pour float16 et bfloat16,
       1 pour int8 ;
    3. cette valeur reste cohérente avec la vérité PyTorch `tensor.element_size()`,
       qui sert d'oracle : si les deux divergent, notre code est faux ;
    4. la conséquence à retenir : un élément FP16 coûte deux fois moins qu'un FP32.

    API à faire émerger (cible roadmap : `src/inference_lab/tensors/dtypes.py`)
    -------------------------------------------------------------------------
        def bytes_per_element(dtype: torch.dtype) -> int: ...

    Indice : la réponse existe déjà dans PyTorch
    (`torch.empty(0, dtype=dtype).element_size()`). L'intérêt du module est de nommer
    le concept une fois pour toute la roadmap et d'échouer explicitement sur un dtype
    non supporté.
    """
    from inference_lab.tensors.dtypes import bytes_per_element

    # Arrange — deux tenseurs de MÊME shape, deux précisions différentes
    fp32 = torch.empty(2, 3, dtype=torch.float32)
    fp16 = torch.empty(2, 3, dtype=torch.float16)

    # Act — ce que notre API doit savoir répondre
    fp32_bytes = bytes_per_element(fp32.dtype)
    fp16_bytes = bytes_per_element(fp16.dtype)

    # Assert 1 — le dtype demandé est bien celui du tenseur
    assert fp32.dtype is torch.float32
    assert fp16.dtype is torch.float16

    # Assert 2 — valeurs attendues, écrites en dur
    assert fp32_bytes == 4
    assert fp16_bytes == 2
    assert bytes_per_element(torch.bfloat16) == 2
    assert bytes_per_element(torch.int8) == 1

    # Assert 3 — cohérence avec l'oracle PyTorch
    assert fp32_bytes == fp32.element_size()
    assert fp16_bytes == fp16.element_size()

    # Assert 4 — la conséquence à retenir
    assert fp32_bytes == 2 * fp16_bytes


@pytest.mark.tdd
def test_reduced_precision_changes_numerical_accuracy():
    """Roadmap 1.8 — moins de bits de mantisse, plus d'erreur d'arrondi.

    Objectif d'apprentissage
    ------------------------
    Réduire la précision n'est jamais gratuit. Le nombre de bits de mantisse fixe
    l'erreur d'arrondi commise en stockant un nombre réel :

        float32  : 24 bits de mantisse, exposant large   -> référence
        float16  : 11 bits de mantisse, exposant étroit  -> précis mais déborde vite
        bfloat16 :  8 bits de mantisse, exposant de FP32 -> moins précis, jamais de débordement

    C'est pourquoi BF16 est préféré en pratique : on accepte de perdre de la précision
    pour garder la plage dynamique de FP32. Comprendre cet arbitrage est indispensable
    avant de comparer notre modèle à la référence Hugging Face avec une tolérance.

    Schéma mental
    -------------
        x (float32)  --.to(fp16)-->  perdu  --.to(float32)-->  x_roundtrip
        erreur = max |x - x_roundtrip|

        1/3 n'est pas représentable exactement en binaire : l'aller-retour laisse
        toujours une trace, plus grande en BF16 (8 bits) qu'en FP16 (11 bits).

    Ce que ce test vérifie
    ----------------------
    1. un aller-retour float32 -> float32 ne perd strictement rien ;
    2. l'aller-retour FP16 perd de l'information, mais reste dans une tolérance
       relative de l'ordre de 1e-3 ;
    3. l'aller-retour BF16 perd PLUS que FP16 sur des valeurs de magnitude proche de 1 ;
    4. l'aller-retour rend un tenseur de nouveau en float32 (on compare toujours dans
       la précision de référence).

    API à faire émerger (cible roadmap : `src/inference_lab/tensors/dtypes.py`)
    -------------------------------------------------------------------------
        def roundtrip(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor: ...
        def max_absolute_error(tensor: torch.Tensor, dtype: torch.dtype) -> float: ...

    Indice : l'aller-retour tient en `tensor.to(dtype).to(tensor.dtype)`, et l'erreur en
    `(tensor - roundtrip(...)).abs().max().item()`. `torch.finfo(dtype)` donne `eps` et
    les bornes de chaque dtype si tu veux vérifier tes ordres de grandeur.
    Attention : ce test isole la perte de MANTISSE. Toutes les valeurs doivent tenir dans la plage
    de FP16, sinon le cast sature à `+inf` et l'assertion 3 devient fausse (FP16 perdrait alors
    infiniment plus que BF16). Le débordement est traité par
    `test_fp16_overflows_where_bf16_keeps_dynamic_range`.
    """

    from inference_lab.tensors.dtypes import max_absolute_error, roundtrip

    # Arrange — un tenseur 1D float32 de fractions NON représentables exactement en binaire (ni
    #           puissances de deux, ni demis), toutes de magnitude proche de 1 pour rester loin des
    #           bornes de plage de FP16 (6.1e-05 .. 65504). Déterministe par construction : aucune
    #           valeur aléatoire, donc aucun seed nécessaire.
    x = torch.tensor([1 / 3, 2 / 3, 0.1, 0.7, 1.234567, 0.987654], dtype=torch.float32)

    # Act — l'erreur absolue maximale de l'aller-retour `x -> dtype -> float32`, pour les trois
    #       précisions.
    error_fp32 = max_absolute_error(x, torch.float32)
    error_fp16 = max_absolute_error(x, torch.float16)
    error_bf16 = max_absolute_error(x, torch.bfloat16)

    # Assert 1 — un aller-retour FP32 -> FP32 ne perd strictement rien
    assert error_fp32 == 0.0

    # Assert 2 — FP16 perd de l'information, mais reste dans ~1e-3 en relatif
    assert error_fp16 > 0.0
    torch.testing.assert_close(roundtrip(x, torch.float16), x, rtol=1e-3, atol=0.0)

    # Assert 3 — BF16 a 8 bits de mantisse contre 11 pour FP16 : il perd davantage
    assert error_bf16 > error_fp16

    # Assert 4 — l'aller-retour ramène bien le dtype de référence
    assert roundtrip(x, torch.float16).dtype is torch.float32


@pytest.mark.tdd
def test_fp16_overflows_where_bf16_keeps_dynamic_range():
    """Roadmap 1.8 — l'exposant décide de la PLAGE, pas de la précision.

    Objectif d'apprentissage
    ------------------------
    Un format flottant a deux budgets indépendants :

        mantisse -> finesse de l'arrondi      exposant -> plage des magnitudes atteignables

        float32  : 24 bits de mantisse, plage +/-3.4e38
        float16  : 11 bits de mantisse, plage +/-65504      <- plage minuscule
        bfloat16 :  8 bits de mantisse, plage +/-3.4e38     <- plage de FP32

    BF16 est mathématiquement MOINS précis que FP16, et pourtant il est préféré en pratique : un
    poids, un gradient ou un logit qui dépasse 65504 devient `+inf` en FP16, et `inf` contamine
    ensuite tout le calcul (`inf * 0 = nan`). Perdre 3 bits de mantisse est réparable, perdre
    l'ordre de grandeur ne l'est pas.

    Schéma mental
    -------------
        989484 --.to(fp16)--> +inf       erreur absolue = inf   (valeur détruite)
        989484 --.to(bf16)--> 991232     erreur absolue = 1748  (ordre de grandeur préservé)

    Ce que ce test vérifie
    ----------------------
    1. les bornes finies de chaque dtype, écrites en dur (65504 pour FP16, > 3e38 pour BF16) ;
    2. exactement un élément du tenseur déborde en FP16, aucun en BF16 ;
    3. le débordement produit `+inf` dans l'aller-retour, donc une erreur absolue infinie : c'est la
       réponse juste, la valeur a été entièrement perdue ;
    4. BF16 reste fini sur ce même tenseur, et l'ordre des erreurs absolues est ici l'INVERSE de
       celui du test de mantisse : comparer FP16 et BF16 par une erreur absolue n'a de sens que si
       personne ne déborde, d'où l'existence de `count_overflows`.

    API à faire émerger (cible roadmap : `src/inference_lab/tensors/dtypes.py`)
    -------------------------------------------------------------------------
        def representable_range(dtype: torch.dtype) -> tuple[float, float]: ...
        def count_overflows(tensor: torch.Tensor, dtype: torch.dtype) -> int: ...

    Indice : `torch.finfo(dtype).min` / `.max` donnent les bornes. Un débordement se détecte en
    comparant la finitude avant et après le cast :
    `tensor.isfinite() & ~tensor.to(dtype).isfinite()`.
    Ne compter que les éléments finis en entrée, et ne jamais renvoyer `nan` depuis une métrique :
    `nan` ne lève pas d'erreur, il rend simplement toutes les comparaisons fausses.
    """

    from inference_lab.tensors.dtypes import (
        count_overflows,
        max_absolute_error,
        representable_range,
        roundtrip,
    )

    # Arrange — un tenseur 1D float32 « riche », chaque élément a un rôle :
    #           1/3 et 0.485785785 -> perte de mantisse seule, dans la plage des deux dtypes ;
    #           37474.0            -> dans la plage FP16, mais déjà grossièrement résolu ;
    #           989484.0           -> HORS plage FP16 (> 65504), dans la plage BF16.
    x_wide = torch.tensor([1 / 3, 0.485785785, 37474.0, 989484.0], dtype=torch.float32)

    # Act — compter les débordements introduits par chaque cast.
    overflows_fp16 = count_overflows(x_wide, torch.float16)
    overflows_bf16 = count_overflows(x_wide, torch.bfloat16)

    # Assert 1 — les bornes finies de chaque dtype
    assert representable_range(torch.float16) == (-65504.0, 65504.0)
    assert representable_range(torch.bfloat16)[1] > 3e38
    # BF16 partage l'exposant de FP32 : sa borne est du même ordre de grandeur, très légèrement
    # inférieure parce qu'avec 8 bits de mantisse la plus grande mantisse représentable est plus
    # grossière (3.3895e38 contre 3.4028e38), pas parce que la plage serait plus étroite.
    assert representable_range(torch.bfloat16)[1] / representable_range(torch.float32)[1] > 0.99
    assert representable_range(torch.bfloat16)[1] > 5e33 * representable_range(torch.float16)[1]

    # Assert 2 — un seul élément déborde en FP16, aucun en BF16
    assert overflows_fp16 == 1
    assert overflows_bf16 == 0

    # Assert 3 — le débordement devient `+inf`, donc une erreur absolue infinie
    assert torch.isinf(roundtrip(x_wide, torch.float16)).sum().item() == 1
    assert math.isinf(max_absolute_error(x_wide, torch.float16))

    # Assert 4 — BF16 conserve l'ordre de grandeur : tout reste fini, et l'erreur est mesurable
    assert torch.isfinite(roundtrip(x_wide, torch.bfloat16)).all()
    assert math.isfinite(max_absolute_error(x_wide, torch.bfloat16))
    assert max_absolute_error(x_wide, torch.bfloat16) < max_absolute_error(x_wide, torch.float16)


@pytest.mark.tdd
def test_fp16_flushes_tiny_values_to_zero_while_absolute_error_hides_it():
    """Roadmap 1.8 — l'autre bout de la plage : les valeurs trop petites tombent à zéro.

    Objectif d'apprentissage
    ------------------------
    L'exposant borne la plage des DEUX côtés. En FP16, le plus petit subnormal vaut 2**-24, soit
    ~5.96e-08 : toute valeur plus petite est écrasée à `0.0`. BF16, avec l'exposant de FP32, descend
    jusqu'à ~1.18e-38 en normalisé.

    Le piège de mesure est ici plus vicieux que le débordement : une valeur de 1e-8 écrasée à zéro
    ne coûte que 1e-8 en erreur ABSOLUE, ce qui passe sous n'importe quelle tolérance, alors que
    l'information est perdue à 100 %. Seule l'erreur RELATIVE le révèle. C'est exactement ce qui
    arrive aux petits gradients ou aux probabilités très faibles d'un softmax.

    Schéma mental
    -------------
        1e-08 --.to(fp16)--> 0.0        erreur absolue = 1e-08 (« négligeable »), relative = 1.0
        1e-08 --.to(bf16)--> ~1e-08     erreur absolue ~ 4e-11,                  relative < 1e-02

    Ce que ce test vérifie
    ----------------------
    1. FP16 écrase 1e-08 à exactement zéro, BF16 non ;
    2. l'erreur absolue reste minuscule et masque donc complètement le problème ;
    3. l'erreur relative vaut exactement 1.0 : toute la valeur a disparu ;
    4. BF16 ne perd que de la mantisse sur ce même tenseur (erreur relative petite).

    API à faire émerger (cible roadmap : `src/inference_lab/tensors/dtypes.py`)
    -------------------------------------------------------------------------
        def max_relative_error(tensor: torch.Tensor, dtype: torch.dtype) -> float: ...

    Indice : `max |x - roundtrip(x)| / |x|`, calculé sur les seuls éléments finis et NON NULS, sans
    quoi la division par zéro fabrique un `nan` qui rend le test faussement vert.
    """

    from inference_lab.tensors.dtypes import max_absolute_error, max_relative_error, roundtrip

    # Arrange — 1e-08 est sous le plus petit subnormal FP16 (2**-24 ~ 5.96e-08) mais très au-dessus
    #           du plus petit normal BF16 (2**-126 ~ 1.18e-38). 0.5 est exact partout et sert de
    #           témoin : il ne contribue à aucune erreur.
    x_tiny = torch.tensor([1e-08, 0.5], dtype=torch.float32)

    # Act — l'aller-retour FP16 et les deux métriques d'erreur sur ce tenseur.
    roundtripped_fp16 = roundtrip(x_tiny, torch.float16)

    # Assert 1 — FP16 écrase la valeur à zéro, BF16 la conserve
    assert roundtripped_fp16[0].item() == 0.0
    assert roundtrip(x_tiny, torch.bfloat16)[0].item() != 0.0

    # Assert 2 — l'erreur absolue déclare le problème « négligeable »
    assert max_absolute_error(x_tiny, torch.float16) <= 1e-08

    # Assert 3 — l'erreur relative dit la vérité : 100 % de la valeur perdue
    assert max_relative_error(x_tiny, torch.float16) == 1.0

    # Assert 4 — BF16 ne perd que de la mantisse
    assert max_relative_error(x_tiny, torch.bfloat16) < 1e-02
    assert max_relative_error(x_tiny, torch.bfloat16) > 0.0


@pytest.mark.tdd
def test_tensor_memory_equals_numel_times_element_size():
    """Roadmap 1.9 — mémoire d'un tenseur = nombre d'éléments x octets par élément.

    Objectif d'apprentissage
    ------------------------
    C'est LA formule que l'on réutilisera partout ensuite : taille des poids du modèle,
    taille du KV cache par token, trafic mémoire d'un kernel. Une fois posée ici sur un
    tenseur de 6 éléments, elle s'applique telle quelle à un tenseur de 500 millions de
    paramètres :

        mémoire (octets) = numel x bytes_per_element(dtype)

    Schéma mental
    -------------
        (2, 3) float32  ->  6 x 4 = 24 octets
        (2, 3) float16  ->  6 x 2 = 12 octets

        La shape ne dit rien du coût mémoire toute seule : il faut la combiner au dtype.

    Ce que ce test vérifie
    ----------------------
    1. les valeurs en octets attendues, calculées à la main (24 et 12) ;
    2. la cohérence avec la formule générale `numel() * element_size()`, valable pour
       n'importe quelle shape ;
    3. la cohérence avec l'allocation réelle observée côté PyTorch
       (`untyped_storage().nbytes()`) pour un tenseur contigu fraîchement alloué ;
    4. à shape identique, FP16 coûte exactement deux fois moins que FP32.

    À noter (section 1.3, pas asserté ici) : une `view` partage le storage de son tenseur
    source, donc elle n'ajoute aucun octet. Cette fonction mesure la taille logique du
    tenseur, pas l'empreinte réelle d'un storage éventuellement partagé.

    API à faire émerger (cible roadmap : `src/inference_lab/tensors/memory.py`)
    -------------------------------------------------------------------------
        def tensor_memory_bytes(tensor: torch.Tensor) -> int: ...

    Indice : `tensor.numel()` et `tensor.element_size()` suffisent. Renvoie un `int`, pas
    un tenseur ni un float, pour que la valeur reste comparable à un calcul fait à la main.
    """


    from inference_lab.tensors.memory import tensor_memory_bytes

    # Arrange — créer `fp32` et `fp16`, deux tenseurs contigus fraîchement alloués de MÊME
    #           shape (2, 3), l'un en `torch.float32`, l'autre en `torch.float16`.

    fp32 = torch.tensor([[0.1,0.02],[3,4],[5,6]],dtype=torch.float32)
    fp16 = torch.tensor([[6,7],[8,9],[10,3]],dtype=torch.float16)

    # Act — demander à l'API la taille en octets de chacun des deux tenseurs.

    # Assert 1 — la formule appliquée à la main, en dur
    assert tensor_memory_bytes(fp32) == 24
    assert tensor_memory_bytes(fp16) == 12

    # Assert 2 — cohérence avec la formule générale, indépendante de la shape
    assert tensor_memory_bytes(fp32) == fp32.numel() * fp32.element_size()

    # Assert 3 — cohérence avec l'allocation réelle du storage
    assert tensor_memory_bytes(fp32) == fp32.untyped_storage().nbytes()

    # Assert 4 — à shape identique, FP16 coûte deux fois moins que FP32
    assert tensor_memory_bytes(fp32) == 2 * tensor_memory_bytes(fp16)
