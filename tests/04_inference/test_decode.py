"""Section 4.5 — le decode : un seul token en entrée, tout le contexte en mémoire.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/04_inference/test_decode.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(entrée (1, 1), logits (1, 1, 16), cache de 3 puis 4 positions, 1 position projetée au lieu
de 4) : c'est volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la
même formule que le code testé.

Le decode est la seconde phase du moteur, celle qui domine la latence perçue. Contrairement
au prefill (4.4), elle est *memory bound* : très peu de calcul (une position) pour beaucoup
d'octets lus (les poids et tout le KV cache). Ces tests fixent son contrat : une position en
entrée, une position ajoutée au cache, zéro recalcul du passé.

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
def test_decode_processes_only_new_token():
    """Roadmap 4.5 — la dimension de séquence en entrée du decode vaut 1.

    Objectif d'apprentissage
    ------------------------
    Une étape de decode ne calcule qu'une chose : la représentation du dernier token. Son
    entrée est donc de shape (batch, 1) et sa sortie (batch, 1, vocab). C'est la différence
    structurelle avec le prefill, qui traite (batch, N).

    Cette asymétrie explique le profil de performance d'un moteur d'inférence : avec une
    seule position, les matmuls dégénèrent en produits matrice-vecteur, le GPU n'est plus
    saturé par le calcul mais par la lecture des poids. C'est ce qui motive le batching
    (section 9) et la quantification (section 10).

    Schéma mental
    -------------
        batch=1, num_layers=2, num_heads=2, num_kv_heads=1, head_dim=4, vocab=16

        cache après prefill : 3 positions
        entrée decode  : next_token_ids (1, 1)         <- UN token, pas 4
        Q du decode    : (1, 2, 1, 4)                  <- 1 requête, 2 têtes
        K/V lus        : (1, 1, 4, 4)                  <- 4 positions, 1 tête KV
        sortie         : logits (1, 1, 16)             <- une seule ligne

    Ce que ce test vérifie
    ----------------------
    1. l'entrée du decode n'a qu'une position : shape (1, 1), dtype entier ;
    2. la sortie n'a qu'une ligne de logits, (1, 1, 16), et non une ligne par position du
       contexte ;
    3. le modèle n'a été appelé qu'une fois, sur une seule position ;
    4. le cache est passé de 3 à 4 positions : la requête est unique, mais elle a bien
       accès aux 4 clés du contexte.

    API à faire émerger (cible roadmap : `src/inference_lab/inference/decode.py`)
    ---------------------------------------------------------------------------
        def decode_step(
            model: torch.nn.Module,
            next_token_ids: torch.Tensor,
            cache: KVCache,
        ) -> torch.Tensor: ...

    Indice : `decode_step` ne reconstruit jamais la séquence complète ; il passe uniquement
    `next_token_ids` au modèle avec le cache. Piège classique : concaténer les ids et
    repasser tout le contexte « pour être sûr » — c'est exactement le decode naïf de 4.1, et
    l'assert sur la shape des logits l'attrape.
    """

    pytest.skip("Roadmap TDD 4.5 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.inference.decode import decode_step

    # Arrange — dimensions jouets constantes dans tout le fichier : batch=1, num_layers=2,
    #           num_heads=2, num_kv_heads=1, head_dim=4, vocab=16, prompt de 3 tokens.
    #           Construire `model`, un modèle causal jouet déterministe (seed fixée) qui
    #           accepte un KV cache, compte ses appels dans `forward_calls` et retient dans
    #           `last_input_length` le nombre de positions reçues ; `cache`, un KVCache déjà
    #           rempli par le prompt (3 positions sur les 2 couches) ; et `next_token_ids`,
    #           l'id du dernier token généré, `torch.long` de shape (1, 1).

    # Act — jouer UNE étape de decode avec `decode_step` et récupérer `logits`.

    # Assert 1 — une seule position en entrée
    assert next_token_ids.shape == (1, 1)
    assert next_token_ids.dtype is torch.long

    # Assert 2 — une seule ligne de logits en sortie
    assert logits.shape == (1, 1, 16)

    # Assert 3 — le modèle n'a vu qu'une position, en un seul appel
    assert model.forward_calls == 1
    assert model.last_input_length == 1

    # Assert 4 — le contexte lu vaut 4 positions, le cache a grandi de 3 à 4
    assert cache.seq_len == 4
    assert cache.get(0)[0].shape == (1, 1, 4, 4)


@pytest.mark.tdd
def test_decode_reuses_cached_keys_and_values():
    """Roadmap 4.5 — les K/V du passé sont relus, jamais recalculés.

    Objectif d'apprentissage
    ------------------------
    C'est la propriété qui rend le cache utile : K et V d'un token ne dépendent que de ce
    token et de sa position, donc ils sont figés dès leur premier calcul. Une étape de
    decode ne projette W_k et W_v que pour la nouvelle position, puis concatène.

    « Relire au lieu de recalculer » se vérifie de deux façons complémentaires, et les deux
    sont dans ce test : l'historique du cache est inchangé bit à bit, et le contenu final du
    cache coïncide avec ce qu'aurait produit une passe complète sur les 4 tokens. Le premier
    assert prouve l'économie, le second prouve la correction.

    Schéma mental
    -------------
        batch=1, num_kv_heads=1, head_dim=4

        avant decode : K (1, 1, 3, 4)   [k0 k1 k2]
        decode t3    : projette k3 SEUL puis concatène
        après decode : K (1, 1, 4, 4)   [k0 k1 k2 k3]
                        ^^^^^^^^ identiques bit à bit

        oracle : une passe complète sur [t0 t1 t2 t3] donne les mêmes 4 clés
                 (à la tolérance float32 près : rtol=1e-5, atol=1e-6)

    Ce que ce test vérifie
    ----------------------
    1. les 3 positions déjà en cache sont strictement inchangées, pour K comme pour V ;
    2. exactement une position a été ajoutée, et son contenu diffère des précédentes ;
    3. le cache obtenu par decode incrémental est numériquement égal à celui d'un recalcul
       complet des 4 tokens ;
    4. le coût : une seule position a été projetée en K/V alors que le contexte en compte
       4 — trois projections économisées pour cette seule étape.

    API à faire émerger (cible roadmap : `src/inference_lab/inference/decode.py`)
    ---------------------------------------------------------------------------
        logits = decode_step(model, next_token_ids, cache)
        cache.get(layer_idx) -> tuple[torch.Tensor, torch.Tensor]

    Indice : `torch.equal` pour l'égalité bit à bit de l'historique, et
    `torch.testing.assert_close` pour la comparaison au recalcul complet — l'ordre des
    opérations diffère entre les deux chemins, l'égalité exacte n'est pas garantie en
    float32. Pense à `.clone()` avant l'appel : sans copie, tu comparerais le cache à
    lui-même et l'assert 1 passerait toujours.
    """

    pytest.skip("Roadmap TDD 4.5 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.inference.decode import decode_step

    # Arrange — mêmes dimensions jouets : batch=1, num_layers=2, num_heads=2, num_kv_heads=1,
    #           head_dim=4, vocab=16, prompt de 3 tokens. Construire `model` (déterministe,
    #           avec un compteur `kv_positions_projected` du nombre de positions pour
    #           lesquelles W_k / W_v ont été appliqués), `cache` déjà rempli par le prompt,
    #           `next_token_ids` de shape (1, 1), puis les COPIES `keys_before` et
    #           `values_before` de la paire (K, V) de la couche 0, shape (1, 1, 3, 4).
    #           Préparer aussi `full_pass_keys`, les 4 clés de la couche 0 obtenues par une
    #           passe complète du même modèle sur les 4 ids (prompt + nouveau token), sans
    #           cache : c'est l'oracle de correction, shape (1, 1, 4, 4).

    # Act — jouer une étape de decode avec `decode_step`, en remettant le compteur
    #       `kv_positions_projected` du modèle à zéro juste avant l'appel.

    # Assert 1 — l'historique est relu, pas réécrit
    assert torch.equal(cache.get(0)[0][:, :, :3, :], keys_before)
    assert torch.equal(cache.get(0)[1][:, :, :3, :], values_before)

    # Assert 2 — une position ajoutée, dont le contenu est nouveau
    assert cache.get(0)[0].shape == (1, 1, 4, 4)
    assert not torch.equal(cache.get(0)[0][:, :, 3, :], cache.get(0)[0][:, :, 2, :])

    # Assert 3 — même résultat qu'un recalcul complet des 4 tokens
    torch.testing.assert_close(cache.get(0)[0], full_pass_keys, rtol=1e-5, atol=1e-6)

    # Assert 4 — une seule position projetée en K/V, au lieu des 4 du contexte
    assert model.kv_positions_projected == 1
    assert cache.seq_len - model.kv_positions_projected == 3
