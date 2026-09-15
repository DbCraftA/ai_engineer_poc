"""Section 15.1 / 15.2 — quantification INT8 des poids : octets gagnés, erreur bornée.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/15_advanced/test_quantization.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(2048 octets en FP16 contre 1024 + 16 octets en INT8 par groupe, échelle 1/127, erreur
maximale 1/254) : c'est volontaire. Un test doit énoncer la vérité attendue, pas la
recalculer avec la même formule que le code testé.

La quantification est le premier compromis assumé de la roadmap : jusqu'ici toutes les
optimisations (KV cache en section 4, GQA en 5.4, SDPA en 8) étaient exactes à la tolérance
flottante près. Ici on accepte de PERDRE de l'information pour diviser les octets, donc il
faut deux tests : un sur la mémoire gagnée (15.1) et un sur l'erreur commise (15.2). Le fil
conducteur est le triplet `valeurs INT8 + échelle + zero_point` : les métadonnées coûtent
des octets et fixent la précision, c'est le même arbitrage que `head_dim` vs `num_kv_heads`.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices. Même raison pour l'import
# du module cible : c'est ton code d'`Act` qui l'appellera, le linter le voit donc inutilisé.
# ruff: noqa: F401, F821

import pytest
import torch


@pytest.mark.tdd
def test_quantized_weights_require_less_storage_than_fp16_weights():
    """Roadmap 15.1 — passer les poids en INT8 divise les octets, presque par deux.

    Objectif d'apprentissage
    ------------------------
    Un poids FP16 coûte 2 octets, un poids INT8 en coûte 1 (section 1.8) : le corps du
    tenseur est divisé par 2 depuis FP16 et par 4 depuis FP32. Mais un INT8 seul ne veut
    rien dire — il faut l'échelle qui le ramène dans le monde réel :

        w ≈ valeur_int8 x scale            (symétrique, zero_point implicite à 0)
        w ≈ (valeur_int8 - zero_point) x scale   (asymétrique)

    Ces métadonnées sont le vrai sujet du test. Une échelle par TENSEUR ne coûte rien mais
    doit encaisser l'outlier le plus violent du tenseur, donc dégrade tous les autres poids.
    Une échelle par GROUPE de 128 coûte quelques octets et isole les outliers dans leur
    groupe. C'est l'arbitrage granularité / précision de tous les formats du marché
    (INT8 par canal, GPTQ ou AWQ en 4 bits par groupe de 128).

    Pour Qwen2.5-0.5B, 500 M de paramètres : 1 Go en FP16, ~500 Mo en INT8 par groupe. Le
    gain n'est pas seulement de la place occupée : au decode, le temps est dominé par la
    lecture des poids depuis la HBM (section 7), donc deux fois moins d'octets à relire,
    c'est presque deux fois plus de tokens par seconde.

    Schéma mental
    -------------
        1024 poids, groupes de 128 -> 8 groupes

            FP32                      : 1024 x 4                    = 4096 octets
            FP16                      : 1024 x 2                    = 2048 octets
            INT8 (corps seul)         : 1024 x 1                    = 1024 octets
            INT8 + 1 échelle FP16     : 1024 + 2                    = 1026 octets
            INT8 + 8 échelles FP16    : 1024 + 8 x 2                = 1040 octets
            + 8 zero_points INT8      : 1024 + 8 x 2 + 8 x 1        = 1048 octets

        2048 / 1040 = 1,969 : le facteur 2 n'est jamais atteint, les échelles se paient

    Ce que ce test vérifie
    ----------------------
    1. les références non quantifiées : 2048 octets en FP16, 4096 en FP32 pour 1024 poids ;
    2. les octets réellement stockés en INT8 : 1024 pour le corps, 1026 avec une échelle par
       tenseur, 1040 avec une échelle par groupe de 128, 1048 si le schéma est asymétrique ;
    3. le gain annoncé : le corps INT8 fait exactement la moitié du FP16 et le quart du FP32,
       mais le total reste au-dessus de la moitié à cause des métadonnées ;
    4. le compromis granularité / précision : 8 échelles au lieu d'une coûtent 14 octets et
       réduisent d'au moins un facteur 4 l'erreur moyenne de quantification.

    API à faire émerger (cible roadmap « future quantization », cible proposée :
    `src/inference_lab/quantization/weights.py`, nouveau package `quantization`)
    ---------------------------------------------------------------------------
        @dataclass
        class QuantizedTensor:
            values: torch.Tensor            # int8
            scales: torch.Tensor            # float16, 1 par tenseur ou 1 par groupe
            zero_points: torch.Tensor | None  # int8, None si symétrique
            group_size: int | None

        def quantize_int8(
            weights: torch.Tensor,
            group_size: int | None = None,
            symmetric: bool = True,
            scale: float | None = None,
        ) -> QuantizedTensor: ...

        def storage_bytes(quantized: QuantizedTensor) -> int: ...

    Indice : `storage_bytes` n'est qu'une somme de `numel() * element_size()` sur les champs
    présents (section 1.9) — c'est justement pour cela qu'il faut le tester : oublier les
    échelles fait annoncer un gain de 2x qui n'existe pas. Piège de `quantize_int8` : la plage
    symétrique est [-127, 127] et non [-128, 127], sinon aucune échelle ne rend l'aller-retour
    symétrique autour de zéro.
    """

    pytest.skip("Roadmap TDD 15.1 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.quantization.weights import quantize_int8, storage_bytes

    # Arrange — `weights_fp16`, un tenseur 1D de 1024 poids en `torch.float16` (seed fixée),
    #           dont la magnitude maximale vaut exactement 1.0 et dont un seul élément, placé
    #           dans le DERNIER groupe de 128, est cet outlier : tous les autres poids sont au
    #           moins 40 fois plus petits en magnitude. `weights_fp32` est le même tenseur
    #           converti en `torch.float32`, pour la référence d'octets.

    # Act — quantifier `weights_fp16` trois fois : une échelle par tenseur (`q_per_tensor`),
    #       une échelle par groupe de 128 (`q_per_group`), et le même groupement en
    #       asymétrique (`q_per_group_asym`) ; relever leurs octets dans `per_tensor_bytes`,
    #       `per_group_bytes` et `per_group_asym_bytes`. Mesurer aussi l'erreur absolue
    #       MOYENNE de l'aller-retour des deux premières variantes dans
    #       `per_tensor_error_mean` et `per_group_error_mean`.

    # Assert 1 — les références non quantifiées, en octets
    assert weights_fp16.numel() == 1024
    assert weights_fp16.numel() * weights_fp16.element_size() == 2048
    assert weights_fp32.numel() * weights_fp32.element_size() == 4096

    # Assert 2 — ce que coûte vraiment l'INT8 : le corps, puis les métadonnées
    assert q_per_tensor.values.dtype is torch.int8
    assert q_per_tensor.values.numel() * q_per_tensor.values.element_size() == 1024
    assert q_per_tensor.zero_points is None
    assert per_tensor_bytes == 1026
    assert per_group_bytes == 1040
    assert q_per_group_asym.zero_points.numel() == 8
    assert per_group_asym_bytes == 1048

    # Assert 3 — le gain : moitié du FP16 sur le corps, un peu moins sur le total
    assert 2 * q_per_group.values.numel() * q_per_group.values.element_size() == 2048
    assert 4 * q_per_group.values.numel() * q_per_group.values.element_size() == 4096
    assert 2048 - per_group_bytes == 1008
    assert per_group_bytes > 1024
    assert 2048 / per_group_bytes == pytest.approx(1.9692, rel=1e-3)
    assert 4096 / per_group_bytes == pytest.approx(3.9385, rel=1e-3)

    # Assert 4 — granularité contre précision : 14 octets de plus, erreur divisée par 4
    assert q_per_tensor.scales.numel() == 1
    assert q_per_group.scales.numel() == 8
    assert per_group_bytes - per_tensor_bytes == 14
    assert float(q_per_tensor.scales.max()) == pytest.approx(1 / 127, rel=1e-3)
    assert float(q_per_group.scales.min()) < (1 / 127) / 10
    assert per_group_error_mean < per_tensor_error_mean / 4


@pytest.mark.tdd
def test_dequantized_output_remains_close_to_reference():
    """Roadmap 15.2 — l'erreur de quantification est bornée par la moitié du pas.

    Objectif d'apprentissage
    ------------------------
    Quantifier puis déquantifier n'est pas un aller-retour exact, mais ce n'est pas non plus
    du hasard : c'est un arrondi sur une grille régulière. En symétrique par tenseur,

        scale = max|x| / 127        q = round(x / scale) borné à [-127, 127]
        x_hat = q x scale           |x - x_hat| <= scale / 2   pour tout x de la grille

    La borne `scale / 2` est la moitié du pas de la grille : elle vient de `round`, pas d'une
    mesure empirique. C'est ce qui rend la quantification utilisable en production — on sait
    à l'avance ce que l'on perd, et la tolérance d'un test se dérive du pas au lieu d'être
    devinée. C'est aussi pourquoi on ne compare JAMAIS un chemin quantifié avec la tolérance
    FP16 de la section 4.6 (rtol=1e-3) : ici l'erreur est ~1e-2 en relatif, et c'est normal.

    La borne a une condition : aucune valeur ne doit sortir de la grille. Un outlier écrasé
    par le `clamp` à 127 n'est plus arrondi, il est TRONQUÉ, et son erreur n'a plus aucune
    limite. C'est exactement le mode de défaillance des schémas trop grossiers, et la raison
    d'être des échelles par groupe de 15.1.

    Schéma mental
    -------------
        x : 1024 valeurs float32, max|x| = 1.0 exactement

            scale       = 1 / 127   ≈ 0,007874
            erreur max <= 1 / 254   ≈ 0,003937        <- borne par construction

        même tenseur avec un outlier de magnitude 50 (échelle recalculée) :
            scale       = 50 / 127  ≈ 0,3937          <- 50x plus grossier
            erreur max <= 50 / 254  ≈ 0,1969          <- borne toujours vraie, mais 50x pire

        même outlier, ancienne échelle CONSERVÉE (outlier écrasé) :
            x_hat[0] = 127 x (1 / 127) = 1.0   ->  erreur = 50 - 1 = 49.0
            49.0 est 12 446 fois la borne : la garantie a disparu

    Ce que ce test vérifie
    ----------------------
    1. le contrat de sortie : valeurs `int8` dont la plus grande magnitude sature à 127,
       aucun `zero_point` en symétrique, déquantifié en float32 de même shape, échelle 1/127 ;
    2. l'aller-retour reste dans la borne dérivée du pas : `rtol=0`, `atol = 1/254`, et
       l'erreur est non nulle (la quantification perd bel et bien de l'information) ;
    3. avec l'outlier intégré à l'échelle, la borne reste vraie mais 50 fois plus lâche :
       l'outlier ne casse pas la garantie, il la dilue pour tous les autres poids ;
    4. avec l'outlier écrasé par le `clamp`, la borne est violée : erreur de 49.0, soit plus
       de 1000 fois `scale / 2`.

    API à faire émerger (cible roadmap « future quantization », cible proposée :
    `src/inference_lab/quantization/weights.py`, nouveau package `quantization`)
    ---------------------------------------------------------------------------
        def quantize_int8(
            weights: torch.Tensor,
            group_size: int | None = None,
            symmetric: bool = True,
            scale: float | None = None,
        ) -> QuantizedTensor: ...

        def dequantize_int8(quantized: QuantizedTensor) -> torch.Tensor: ...

        Le paramètre `scale` sert à FORCER une échelle déjà calculée : c'est ce qui permet de
        rejouer une quantification calibrée sur d'autres données, donc de provoquer la
        saturation de l'assert 4.

    Indice : `torch.round` arrondit au plus proche (moitiés vers le pair), c'est lui qui donne
    la borne `scale / 2` ; `torch.clamp(..., -127, 127)` est ce qui la casse quand une valeur
    dépasse la grille. Ajoute une petite marge à `atol` (1e-6) : le quotient `x / scale` est
    calculé en float32, l'égalité au pas exact se joue au dernier bit. Piège : `.to(torch.int8)`
    TRONQUE vers zéro, il ne remplace pas `round` — sans arrondi explicite, l'erreur double.
    """

    pytest.skip("Roadmap TDD 15.2 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.quantization.weights import dequantize_int8, quantize_int8

    # Arrange — `x`, un tenseur 1D de 1024 valeurs en `torch.float32` (seed fixée), non
    #           représentables exactement en binaire, normalisé pour que `x.abs().max()` vaille
    #           exactement 1.0. Puis `x_outlier`, une copie de `x` dont l'élément d'INDICE 0
    #           est remplacé par un outlier POSITIF valant 50 fois le maximum des autres.

    # Act — quantifier `x` en INT8 symétrique par tenseur (`q`, échelle relevée dans `scale`)
    #       et le déquantifier (`dequantized`, erreur absolue maximale dans `error_max`).
    #       Quantifier ensuite `x_outlier` en laissant l'API recalculer l'échelle
    #       (`q_outlier`, `outlier_scale`, `outlier_error_max`), puis une dernière fois en
    #       FORÇANT l'échelle `scale` du premier appel (`q_saturated`,
    #       `dequantized_saturated`, `saturated_error_max`).

    # Assert 1 — le contrat de sortie et l'échelle attendue
    assert q.values.dtype is torch.int8
    assert int(q.values.abs().max()) == 127
    assert q.zero_points is None
    assert dequantized.shape == (1024,)
    assert dequantized.dtype is torch.float32
    assert scale == pytest.approx(1 / 127, rel=1e-6)

    # Assert 2 — l'aller-retour tient dans la moitié du pas, et perd quelque chose
    torch.testing.assert_close(dequantized, x, rtol=0, atol=1 / 254 + 1e-6)
    assert error_max > 0.0
    assert error_max <= 1 / 254 + 1e-6

    # Assert 3 — l'outlier absorbé par l'échelle : borne conservée, 50 fois plus lâche
    assert outlier_scale == pytest.approx(50 / 127, rel=1e-6)
    assert outlier_scale == pytest.approx(50 * scale, rel=1e-6)
    assert outlier_error_max <= 50 / 254 + 1e-6
    assert outlier_error_max > 1 / 254

    # Assert 4 — l'outlier écrasé par le clamp : plus aucune borne
    assert int(q_saturated.values[0]) == 127
    assert float(dequantized_saturated[0]) == pytest.approx(1.0, rel=1e-6)
    assert saturated_error_max == pytest.approx(49.0, rel=1e-6)
    assert saturated_error_max > 1000 * (1 / 254)
