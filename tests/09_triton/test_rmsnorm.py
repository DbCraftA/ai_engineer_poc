"""Section 9.4 — RMSNorm en Triton : un kernel fusionné validé par la référence PyTorch.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/09_triton/test_rmsnorm.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(shape (1, 4, 896), RMS de sortie = 1, poids x2 -> sortie x2, tolérances rtol=1e-5 /
atol=1e-5) : c'est volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec
la même formule que le code testé.

Cap du chapitre : ce kernel n'existe pas pour « aller plus vite » mais pour répondre à une
question mémoire précise. RMSNorm ne coûte presque aucun FLOP, pourtant en PyTorch il relit et
réécrit les activations plusieurs fois (carré, moyenne, rsqrt, produit) : il est limité par la
bande passante. Le fusionner en un seul passage — une lecture, une écriture — est le gain
visé. Le test, lui, ne mesure rien : il vérifie la CORRECTION contre `rms_norm`, la référence
PyTorch de la section 2.11, qui joue le rôle d'oracle. La section 2.11 doit donc être verte
avant celle-ci, et la formule reste strictement la même : mêmes noms, mêmes arguments.

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
def test_triton_rmsnorm_matches_pytorch_reference():
    """Roadmap 9.4 — un kernel fusionné n'a le droit de changer que le nombre de passages mémoire.

    Objectif d'apprentissage
    ------------------------
    C'est le premier kernel « utile » du chapitre : il remplace une opération réelle du modèle
    Qwen2.5-0.5B, appliquée deux fois par bloc de transformer. Le motif est un program par
    token : le program charge la ligne entière de 896 valeurs, calcule sa moyenne des carrés,
    applique `1/sqrt(mean + eps)`, multiplie par le poids et écrit la ligne. Tout tient dans les
    registres : les intermédiaires ne redescendent jamais en mémoire globale, ce qui est
    exactement le gain recherché sur une opération limitée par la bande passante.

    Deux pièges apparaissent avec ce motif. `hidden = 896` n'est pas une puissance de deux : le
    bloc doit être arrondi à 1024, donc 128 voies sont masquées — et elles doivent être chargées
    avec `other=0.0`, sinon la somme des carrés est gonflée et la RMS de sortie ne vaut plus 1.
    Ensuite la division par `hidden` doit utiliser 896, la vraie longueur de ligne, jamais la
    taille de bloc. Un kernel qui se trompe là reste plausible visuellement : seule la
    comparaison à l'oracle PyTorch le démasque.

    Schéma mental
    -------------
        x (batch=1, seq=4, hidden=896), weight (896,), eps = 1e-6
        4 lignes -> grille de 4 programs, BLOCK_SIZE = 1024 (896 arrondi à la puissance de 2)

        program r : offs = [0..1023], mask = offs < 896  (128 voies masquées, other=0.0)
                    mean = sum(x^2) / 896
                    y = x * (1 / sqrt(mean + eps)) * weight

        réduction sur 896 termes -> ordre différent de PyTorch -> écart ~1e-7, pas 0
        weight = ones -> sqrt(mean(y^2)) == 1 pour chacun des 4 tokens
        weight x 2    -> sortie x 2 (le poids est un simple facteur par canal)

    Ce que ce test vérifie
    ----------------------
    1. la shape (1, 4, 896), le dtype float32, le device `"cuda"` et la contiguïté sont
       conservés : la normalisation ne change pas la géométrie du tenseur ;
    2. la sortie correspond à `rms_norm` de la section 2.11, l'oracle, avec rtol=1e-5 et
       atol=1e-5 : une réduction est en jeu, donc une tolérance et non `torch.equal` ;
    3. avec un poids unitaire, la moyenne quadratique de chaque token de sortie vaut 1, ce qui
       prouve au passage que les 128 voies masquées ont bien été chargées à 0.0 ;
    4. le poids agit canal par canal et multiplicativement : le doubler double la sortie, et
       aucun NaN n'apparaît.

    API à faire émerger (cible roadmap : `src/inference_lab/kernels/triton/rmsnorm.py`)
    ---------------------------------------------------------------------------------
        def triton_rms_norm(
            x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
        ) -> torch.Tensor: ...

    Indice : la référence à reproduire est
    `x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight` (section 2.11) ;
    côté kernel, `tl.sum(x * x, axis=0) / hidden` puis `tl.rsqrt(mean + eps)`, et
    `BLOCK_SIZE = triton.next_power_of_2(hidden)` en `tl.constexpr`. Pièges : placer `eps` hors
    de la racine (`sqrt(mean) + eps`), aplatir le tenseur en une seule ligne au lieu de
    normaliser token par token, et charger le poids sans masque.
    """

    pytest.skip("Roadmap TDD 9.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.kernels.triton.rmsnorm import triton_rms_norm
    from inference_lab.nn.normalization.rmsnorm import rms_norm

    # Arrange — batch=1, seq=4, hidden=896 (la dimension cachée de Qwen2.5-0.5B). Appeler
    #           `torch.manual_seed(0)` d'abord, puis créer, tous en `torch.float32`, contigus et
    #           alloués directement sur `"cuda"` : `x` de shape (1, 4, 896), valeurs de
    #           magnitude proche de 1, positives ET négatives, aucun token entièrement nul ;
    #           `weight` de shape (896,) aux valeurs distinctes et non nulles ; `w_ones` de
    #           shape (896,) rempli de 1.0 ; `eps`, le flottant 1e-6, passé explicitement à
    #           chaque appel des deux implémentations.

    # Act — calculer `out = triton_rms_norm(x, weight, eps)`, la référence PyTorch
    #       `reference = rms_norm(x, weight, eps)`, puis `out_ones = triton_rms_norm(x, w_ones,
    #       eps)` et `out_double`, obtenu en appelant le kernel avec un poids valant 2 * `w_ones`.

    # Assert 1 — normalisation « sur place » au sens de la géométrie du tenseur
    assert out.shape == (1, 4, 896)
    assert out.dtype is torch.float32
    assert out.device.type == "cuda"
    assert out.is_contiguous()

    # Assert 2 — l'oracle est la référence PyTorch de la section 2.11, à la tolérance fp32
    torch.testing.assert_close(out, reference, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        triton_rms_norm(x, weight, eps), rms_norm(x, weight, eps), rtol=1e-5, atol=1e-5
    )

    # Assert 3 — poids unitaire : RMS de sortie = 1 par token, donc masquage à 0.0 correct
    torch.testing.assert_close(
        out_ones.pow(2).mean(dim=-1).sqrt(),
        torch.ones(1, 4, device="cuda"),
        rtol=1e-4,
        atol=1e-4,
    )

    # Assert 4 — le poids est un facteur par canal, et le kernel ne produit aucun NaN
    torch.testing.assert_close(out_double, 2.0 * out_ones, rtol=1e-5, atol=1e-5)
    assert not bool(out.isnan().any())
