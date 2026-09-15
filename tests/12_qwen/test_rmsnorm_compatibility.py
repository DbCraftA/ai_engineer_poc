"""Section 12.2 — notre RMSNorm doit être numériquement celle de Qwen2.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/12_qwen/test_rmsnorm_compatibility.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont ici fournies par la
référence Hugging Face `Qwen2RMSNorm` avec les MÊMES poids que les nôtres, et les tolérances
sont écrites en dur (fp32 : rtol=1e-5, atol=1e-6 ; fp16 : 1e-3) : c'est volontaire. Un test
doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Dimensions de la section 12 en mode « sans poids réels » : batch=1, seq=4, hidden=16,
num_heads=4, head_dim=4, num_kv_heads=2, intermediate=32, vocab=32. La référence est un module
Hugging Face isolé, construit sur une `Qwen2Config` réduite avec des poids aléatoires seedés :
c'est exact, local, rapide et déterministe, aucun checkpoint n'est téléchargé. Le vrai
`Qwen/Qwen2.5-0.5B-Instruct` est réservé aux tests 12.6 et 12.7.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
@pytest.mark.model
@pytest.mark.hf
def test_our_rmsnorm_matches_qwen_reference():
    """Roadmap 12.2 — même formule, même epsilon, et surtout même précision de calcul.

    Objectif d'apprentissage
    ------------------------
    C'est le premier test de compatibilité de la section : il fixe la méthode que l'on
    réutilisera pour la RoPE, la GQA, le MLP puis le modèle entier. On ne compare pas deux
    implémentations « à peu près », on impose des poids IDENTIQUES des deux côtés et une
    tolérance justifiée. RMSNorm encadre chaque sous-couche des 24 blocs de Qwen2.5 : un écart
    de 1e-3 ici devient un token de sortie différent après 24 couches.

    Le piège du chapitre est de précision, pas d'algèbre : Qwen (comme Llama) calcule la
    variance et la division en float32 MÊME quand les activations sont en fp16 ou bf16, puis
    revient au dtype d'entrée avant de multiplier par le poids. Sans cet upcast, `x.pow(2)`
    dépasse la plage de fp16 (max ~65504) dès que les activations valent quelques centaines :
    la moyenne devient `inf`, `rsqrt(inf)` vaut 0, et la sortie est un tenseur de zéros
    parfaitement finis et parfaitement faux.

    Schéma mental
    -------------
        x (batch=1, seq=4, hidden=16), weight (16,), eps = 1e-6

        Qwen2RMSNorm.forward :
            x --.to(float32)--> var = mean(x^2) --> x * rsqrt(var + eps)
              --.to(dtype d'entrée)--> * weight   ->  sortie (1, 4, 16), dtype d'entrée

        fp32, magnitude ~1   : notre sortie == référence à 1e-6 près
        fp16, magnitude ~300 : x^2 ~ 90000 > 65504
            sans upcast -> mean = inf -> rsqrt = 0 -> sortie = zéros (faux)
            avec upcast -> sortie correcte, dtype float16 conservé

    Ce que ce test vérifie
    ----------------------
    1. en float32, notre sortie est celle de `Qwen2RMSNorm` à rtol=1e-5 / atol=1e-6, à shape
       et dtype conservés ;
    2. `eps` est bien le `rms_norm_eps` de la config et il est SOUS la racine : un eps
       grossièrement différent fait divorcer les deux implémentations ;
    3. en float16 sur des activations de grande magnitude, notre sortie reste celle de la
       référence (tolérance fp16 1e-3), en float16, non nulle : l'upcast en float32 a bien eu
       lieu ;
    4. la normalisation est indépendante du contexte : normaliser le seul token 2 donne la même
       ligne que dans la sortie complète (c'est ce qui rend RMSNorm compatible avec le decode).

    API à faire émerger (cible roadmap : « Qwen components », cible proposée : réutilisation de
    `src/inference_lab/nn/normalization/rmsnorm.py` de la section 2.11)
    ------------------------------------------------------------------
        def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor: ...

    Indice : aucun nouveau module à écrire si la section 2.11 est déjà verte — il suffit
    d'ajouter l'upcast : calculer la variance sur `x.to(torch.float32)`, puis revenir au dtype
    d'entrée AVANT la multiplication par le poids, exactement dans cet ordre. Le code de
    référence tient en trois lignes et se lit avec
    `inspect.getsource(Qwen2RMSNorm.forward)` si tu veux vérifier ton implémentation.
    """

    pytest.skip("Roadmap TDD 12.2 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.normalization.rmsnorm import rms_norm

    # Arrange — référence Hugging Face locale, sans checkpoint
    #           (`from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm`).
    #           batch=1, seq=4, hidden=16, `eps` = 1e-6 (le `rms_norm_eps` de Qwen2.5).
    #           `x` : tenseur (1, 4, 16) en `torch.float32`, déterministe (`torch.manual_seed(0)`),
    #           de magnitude proche de 1, valeurs positives ET négatives.
    #           `hf_norm` : un `Qwen2RMSNorm(16, eps=eps)` dont le poids a été remplacé par des
    #           valeurs distinctes et non unitaires, seedées.
    #           `weight` : EXACTEMENT le poids de `hf_norm` (`hf_norm.weight.detach()`), sinon la
    #           comparaison ne veut rien dire.
    #           `x_fp16` : le même `x` multiplié par 300 puis converti en `torch.float16`, de
    #           sorte que `x^2` sorte de la plage de fp16.
    #           `hf_norm_fp16` : une copie de `hf_norm` convertie en float16, et `weight_fp16`
    #           son poids.

    # Act — sous `torch.no_grad()`, calculer notre `out = rms_norm(x, weight, eps)` et la
    #       référence `out_hf = hf_norm(x)` ; puis la même paire en float16 :
    #       `out_fp16 = rms_norm(x_fp16, weight_fp16, eps)` et `out_hf_fp16 = hf_norm_fp16(x_fp16)`.

    # Assert 1 — égalité numérique en float32, la tolérance de référence du dépôt
    assert out.shape == (1, 4, 16)
    assert out.dtype is torch.float32
    torch.testing.assert_close(out, out_hf, rtol=1e-5, atol=1e-6)

    # Assert 2 — eps vient de la config et se place sous la racine
    assert not torch.allclose(rms_norm(x, weight, 1.0), out_hf, rtol=1e-4, atol=1e-4)

    # Assert 3 — fp16 : l'upcast en float32 est obligatoire, sinon la sortie s'effondre à zéro
    assert out_fp16.dtype is torch.float16
    assert bool(out_fp16.isfinite().all())
    assert bool((out_fp16 != 0).any())
    torch.testing.assert_close(out_fp16, out_hf_fp16, rtol=1e-3, atol=1e-3)

    # Assert 4 — un token se normalise sans connaître les autres
    torch.testing.assert_close(
        rms_norm(x[:, 2:3], weight, eps), out_hf[:, 2:3], rtol=1e-5, atol=1e-6
    )
