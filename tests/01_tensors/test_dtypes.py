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
    """

    pytest.skip("Roadmap TDD 1.8 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.tensors.dtypes import max_absolute_error, roundtrip

    # Arrange — créer `x`, un petit tenseur 1D en `torch.float32` dont les valeurs ne sont
    #           PAS représentables exactement en binaire (des fractions, pas des puissances
    #           de deux ni des demis), de magnitudes proches de 1 pour rester loin des
    #           limites de plage de FP16. Test déterministe : pas de valeurs aléatoires,
    #           ou alors `torch.manual_seed(0)`.
    torch.manual_seed(0)
    data = [[89,28,9094094,37474,989484,49849894]]
    x = torch.tensor(data,dtype=torch.float32)

    # Act — faire l'aller-retour `x -> dtype -> float32` pour float32, float16 puis bfloat16,
    #       et mesurer l'erreur absolue maximale de chaque aller-retour.
    y = x.to(torch.float16)
    x.to(torch.float32)

    # Assert 1 — un aller-retour FP32 -> FP32 ne perd strictement rien
    assert max_absolute_error(x, torch.float32) == 0.0

    # Assert 2 — FP16 perd de l'information, mais reste dans ~1e-3 en relatif
    assert max_absolute_error(x, torch.float16) > 0.0
    torch.testing.assert_close(roundtrip(x, torch.float16), x, rtol=1e-3, atol=0.0)

    # Assert 3 — BF16 a 8 bits de mantisse contre 11 pour FP16 : il perd davantage
    assert max_absolute_error(x, torch.bfloat16) > max_absolute_error(x, torch.float16)

    # Assert 4 — l'aller-retour ramène bien le dtype de référence
    assert roundtrip(x, torch.float16).dtype is torch.float32


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

    pytest.skip("Roadmap TDD 1.9 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.tensors.memory import tensor_memory_bytes

    # Arrange — créer `fp32` et `fp16`, deux tenseurs contigus fraîchement alloués de MÊME
    #           shape (2, 3), l'un en `torch.float32`, l'autre en `torch.float16`.

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
