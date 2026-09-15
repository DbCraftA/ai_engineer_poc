"""Section 2.7 — attention multi-têtes : découper puis recoller la dimension cachée.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/02_attention/test_multi_head_attention.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
((1, 2, 4, 4), 8 = 2 x 4, 32 éléments) : c'est volontaire. Un test doit énoncer la vérité
attendue, pas la recalculer avec la même formule que le code testé.

Dimensions constantes de la section : batch=1, seq=4, hidden=8, num_heads=2, head_dim=4.
Le multi-têtes ne change ni le nombre de paramètres ni le nombre d'éléments : c'est une
réorganisation de la mémoire (`view` + `transpose`) qui permet à plusieurs sous-espaces
d'attendre des choses différentes. C'est aussi la base des layouts du KV cache (section 4).

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_hidden_dimension_is_split_across_attention_heads():
    """Roadmap 2.7 — hidden = num_heads x head_dim, découpé sans copier de données.

    Objectif d'apprentissage
    ------------------------
    Une tête d'attention n'est pas une projection supplémentaire : c'est une TRANCHE de la
    dimension cachée. On garde les mêmes matrices Q/K/V de taille (hidden, hidden), puis on
    relit le résultat comme `num_heads` sous-vecteurs de taille `head_dim`. D'où la relation
    non négociable `hidden == num_heads * head_dim` (Qwen2.5-0.5B : 896 = 14 x 64). Comme le
    découpage est un simple changement de vue, le nombre d'éléments et le coût mémoire sont
    inchangés.

    Schéma mental
    -------------
        x (batch=1, seq=4, hidden=8)
          --view(1, 4, num_heads=2, head_dim=4)-->  (1, 4, 2, 4)
          --transpose(1, 2)---------------------->  (1, 2, 4, 4) = (batch, heads, seq, head_dim)

        tête 0 du token 0 = x[0, 0, 0:4]      tête 1 du token 0 = x[0, 0, 4:8]
        8 elements x 4 tokens = 32 elements, avant comme après

    Ce que ce test vérifie
    ----------------------
    1. la shape après découpage est (batch, heads, seq, head_dim) = (1, 2, 4, 4), avec
       8 = 2 x 4 ;
    2. la tête n contient bien la tranche contiguë correspondante de la dimension cachée ;
    3. le nombre d'éléments est inchangé (32) : c'est une réorganisation, pas un calcul ;
    4. une configuration impossible (hidden non divisible par num_heads) échoue explicitement.

    API à faire émerger (cible roadmap : `src/inference_lab/nn/attention/mha.py`)
    ---------------------------------------------------------------------------
        def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor: ...

    Indice : `x.view(batch, seq, num_heads, head_dim).transpose(1, 2)` ; `head_dim` se déduit
    de `hidden // num_heads`. Piège : `view(batch, num_heads, seq, head_dim)` directement
    donne la bonne shape mais mélange tokens et têtes — l'assert 2 le détecte. Lève une
    `ValueError` si `hidden % num_heads != 0`.
    """

    pytest.skip("Roadmap TDD 2.7 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.attention.mha import split_heads

    # Arrange — batch=1, seq=4, hidden=8, num_heads=2, head_dim=4.
    #           `x` : tenseur (1, 4, 8) en `torch.float32` dont les 32 valeurs sont TOUTES
    #           DISTINCTES (sinon l'assert 2 ne prouve rien), déterministe.

    # Act — appeler `split_heads(x, 2)` et nommer le résultat `heads`.

    # Assert 1 — (batch, heads, seq, head_dim), avec hidden = 2 x 4
    assert heads.shape == (1, 2, 4, 4)
    assert x.shape[-1] == 2 * 4

    # Assert 2 — chaque tête est une tranche contiguë de la dimension cachée
    torch.testing.assert_close(heads[0, 0, 0], x[0, 0, 0:4], atol=0.0, rtol=0.0)
    torch.testing.assert_close(heads[0, 1, 0], x[0, 0, 4:8], atol=0.0, rtol=0.0)
    torch.testing.assert_close(heads[0, 1, 3], x[0, 3, 4:8], atol=0.0, rtol=0.0)

    # Assert 3 — aucune donnée créée ni perdue : 1 x 4 x 8 = 32 éléments
    assert heads.numel() == 32
    assert heads.numel() == x.numel()

    # Assert 4 — 8 n'est pas divisible par 3 : la configuration doit être refusée
    with pytest.raises(ValueError):
        split_heads(x, 3)


@pytest.mark.tdd
def test_attention_heads_are_concatenated_back_to_hidden_dimension():
    """Roadmap 2.7 — recoller les têtes rend exactement la shape d'entrée.

    Objectif d'apprentissage
    ------------------------
    Après avoir attentionné tête par tête, il faut revenir à un vecteur unique par token pour
    la projection de sortie puis le résidu : les têtes sont concaténées le long de `hidden`,
    dans l'ORDRE des têtes. Cette opération est l'inverse exact du découpage, ce qui donne un
    test en aller-retour très solide. C'est aussi le point où l'on rencontre pour de bon la
    contrainte de contiguïté : après `transpose`, la mémoire n'est plus dans l'ordre attendu
    par `view`.

    Schéma mental
    -------------
        heads (1, num_heads=2, seq=4, head_dim=4)
          --transpose(1, 2)--> (1, 4, 2, 4)  (non contigu)
          --contiguous()-----> (1, 4, 2, 4)
          --view(1, 4, 8)----> (1, 4, 8) = shape d'entrée

        token 0 : [ tête 0 (4 valeurs) | tête 1 (4 valeurs) ] = 8 valeurs
        split_heads puis merge_heads == identité (bit à bit)

    Ce que ce test vérifie
    ----------------------
    1. la concaténation rend exactement la shape d'entrée (1, 4, 8) ;
    2. l'aller-retour découpage / concaténation est l'identité, valeur par valeur ;
    3. l'ordre des têtes compte : intervertir les têtes change le résultat ;
    4. le tenseur recollé est contigu, alors qu'une vue transposée ne l'est pas et se voit
       refuser un `view`.

    API à faire émerger (cible roadmap : `src/inference_lab/nn/attention/mha.py`)
    ---------------------------------------------------------------------------
        def merge_heads(heads: torch.Tensor) -> torch.Tensor: ...

    Indice : `heads.transpose(1, 2).contiguous().view(batch, seq, num_heads * head_dim)`, ou
    `reshape` qui gère la copie tout seul. Piège : `view` juste après `transpose` lève
    « view size is not compatible with input tensor's size and stride » — le message à savoir
    reconnaître (section 1.6).
    """

    pytest.skip("Roadmap TDD 2.7 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.attention.mha import merge_heads, split_heads

    # Arrange — batch=1, seq=4, hidden=8, num_heads=2, head_dim=4.
    #           `x` : tenseur (1, 4, 8) en `torch.float32` aux 32 valeurs toutes distinctes,
    #           déterministe.
    #           `heads` : le résultat de `split_heads(x, 2)`, de shape (1, 2, 4, 4).
    #           `heads_swapped` : les mêmes données avec les deux têtes interverties (par
    #           exemple `heads.flip(1)`).

    # Act — calculer `merged = merge_heads(heads)`, puis `merged_swapped` en appelant
    #       `merge_heads` sur `heads_swapped`.

    # Assert 1 — on retrouve la shape d'entrée, prête pour la projection de sortie
    assert merged.shape == (1, 4, 8)

    # Assert 2 — aller-retour = identité exacte, aucune valeur déplacée
    assert torch.equal(merged, x)
    assert torch.equal(merge_heads(split_heads(x, 2)), x)

    # Assert 3 — l'ordre des têtes fait partie du contrat
    assert not torch.equal(merged_swapped, x)
    torch.testing.assert_close(merged_swapped[0, 0, 0:4], x[0, 0, 4:8], atol=0.0, rtol=0.0)

    # Assert 4 — la sortie est contiguë, alors qu'une vue transposée ne l'est pas
    assert merged.is_contiguous()
    assert not x.transpose(1, 2).is_contiguous()
    with pytest.raises(RuntimeError):
        x.transpose(1, 2).view(1, 4, 8)
