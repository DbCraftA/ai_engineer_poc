"""Section 4.1 — pourquoi le decode naïf coûte de plus en plus cher à chaque token.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/04_inference/test_naive_decode.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(3 puis 4 puis 5 positions traitées, 12 au total au lieu de 5) : c'est volontaire.
Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule que le
code testé.

Cette section installe la MOTIVATION du KV cache : sans cache, générer le token n+1
demande de réencoder les n tokens précédents, donc de recalculer leurs K et V. Le coût
d'une génération devient quadratique en longueur de séquence. Les sections 4.2 à 4.8
construisent le cache qui supprime ce gâchis ; ici on se contente de le MESURER.

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
def test_naive_decode_reprocesses_previous_tokens():
    """Roadmap 4.1 — sans cache, chaque nouveau token relit toute la séquence déjà générée.

    Objectif d'apprentissage
    ------------------------
    Un décodeur autorégressif produit UN token par passe. Si la passe repart des ids
    complets, le modèle reprojette Q, K et V pour tous les tokens précédents alors que
    leurs K/V sont identiques d'une étape à l'autre : ils ne dépendent que du token et de
    sa position, jamais des tokens futurs. Le travail utile d'une étape de decode est donc
    une seule position ; tout le reste est du recalcul.

    C'est la raison d'être du KV cache et de la séparation prefill / decode : sur
    Qwen2.5-0.5B avec un prompt de 1 000 tokens, le decode naïf refait 1 000 positions
    par token généré au lieu d'une seule. Avant d'optimiser, on rend le gâchis mesurable.

    Schéma mental
    -------------
        prompt de 3 tokens, 3 tokens à générer (batch=1, num_layers=2, num_heads=2,
        num_kv_heads=1, head_dim=4, vocab=16)

        étape 1 : [t0 t1 t2]          -> 3 positions -> t3
        étape 2 : [t0 t1 t2 t3]       -> 4 positions -> t4   (t0..t2 recalculés)
        étape 3 : [t0 t1 t2 t3 t4]    -> 5 positions -> t5   (t0..t3 recalculés)
                                         --
        total naïf   : 3 + 4 + 5 = 12 positions
        total avec KV cache : 3 + 1 + 1 = 5 positions

    Ce que ce test vérifie
    ----------------------
    1. le decode naïf appelle le modèle exactement une fois par token généré (3 appels)
       et rend le prompt suivi des 3 nouveaux tokens, soit 6 ids ;
    2. chaque passe traite TOUS les tokens déjà connus : 3, puis 4, puis 5 positions ;
    3. la croissance est linéaire : exactement une position de plus à chaque étape ;
    4. le coût total est de 12 positions au lieu de 5 avec un cache, soit 7 recalculs
       strictement inutiles.

    API à faire émerger (cible roadmap « instrumentation inference »,
    cible proposée : `src/inference_lab/inference/naive.py`)
    ----------------------------------------------------------------
        def naive_decode(
            model: torch.nn.Module,
            prompt_ids: torch.Tensor,
            max_new_tokens: int,
        ) -> tuple[torch.Tensor, list[int]]: ...

    La fonction rend `(output_ids, positions_per_step)` : les ids complets et le nombre de
    positions passées au modèle à chaque étape. Cette liste est l'instrument de mesure.

    Indice : la boucle naïve tient en trois lignes — `logits = model(ids)`, puis
    `next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)`, puis
    `ids = torch.cat([ids, next_id], dim=1)`. Le piège est de mesurer le temps : sur des
    dimensions jouets il est dominé par le bruit. Compte des positions, pas des secondes.
    """

    pytest.skip("Roadmap TDD 4.1 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.inference.naive import naive_decode

    # Arrange — dimensions jouets constantes dans tout le fichier : batch=1, num_layers=2,
    #           num_heads=2, num_kv_heads=1, head_dim=4 (donc hidden=8), vocab=16, prompt de
    #           3 tokens. Construire `prompt_ids`, un tenseur d'ids `torch.long` de shape
    #           (1, 3), et `model`, un modèle causal jouet DÉTERMINISTE (`torch.manual_seed`
    #           fixée) qui rend des logits de shape (batch, seq, 16) et qui expose un
    #           compteur `forward_calls` incrémenté à chaque appel de `forward`.

    # Act — générer 3 nouveaux tokens avec `naive_decode`, et récupérer `output_ids` ainsi
    #       que `positions_per_step`, la liste des longueurs de séquence vues par le modèle.

    # Assert 1 — une passe de modèle par token généré, et 3 + 3 = 6 ids en sortie
    assert model.forward_calls == 3
    assert output_ids.shape == (1, 6)
    assert output_ids.dtype is torch.long

    # Assert 2 — chaque passe réencode tout le contexte déjà connu
    assert positions_per_step == [3, 4, 5]

    # Assert 3 — croissance linéaire : +1 position par étape
    assert positions_per_step[1] - positions_per_step[0] == 1
    assert positions_per_step[2] - positions_per_step[1] == 1

    # Assert 4 — le gâchis chiffré : 12 positions traitées là où un cache en demanderait 5
    assert sum(positions_per_step) == 12
    assert sum(positions_per_step) - 5 == 7
