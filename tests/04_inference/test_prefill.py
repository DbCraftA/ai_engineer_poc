"""Section 4.4 — le prefill : traiter tout le prompt en une seule passe.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/04_inference/test_prefill.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(logits (1, 3, 16), un seul appel de modèle, cache de longueur 3, 192 octets) : c'est
volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même
formule que le code testé.

Le prefill est la première des deux phases d'un moteur d'inférence. Il voit tous les
tokens du prompt d'un coup, ce qui le rend *compute bound* (grosses matrices, GPU bien
occupé), et il laisse derrière lui le KV cache construit en 4.2 / 4.3, à la longueur exacte
du prompt. La phase decode (4.5) part de cet état.

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
def test_prefill_processes_complete_prompt():
    """Roadmap 4.4 — les N tokens du prompt passent dans le modèle en une seule fois.

    Objectif d'apprentissage
    ------------------------
    Le prompt est connu en entier avant de générer quoi que ce soit : rien n'oblige à le
    parcourir token par token. Une passe unique sur (1, 3) exploite le parallélisme des
    matmuls et le masque causal garantit qu'aucun token ne voit son futur — le résultat est
    donc identique à 3 passes successives, pour un coût bien moindre.

    Le prefill produit un vecteur de logits PAR position, mais une seule ligne sert à
    générer : la dernière. Les autres lignes n'existent que parce que le calcul est
    vectorisé (elles servent en revanche à calculer la loss pendant l'entraînement).

    Schéma mental
    -------------
        batch=1, num_layers=2, num_heads=2, num_kv_heads=1, head_dim=4, hidden=8, vocab=16

        prompt_ids (1, 3) --1 seule passe--> logits (1, 3, 16)
                                                       ^
                                       logits[:, -1, :] (1, 16) -> argmax -> 1er token généré

        appels de modèle : 1 (et non 3)

    Ce que ce test vérifie
    ----------------------
    1. une seule passe de modèle traite les 3 positions du prompt ;
    2. les logits renvoyés ont la shape (batch=1, seq=3, vocab=16) : un score par token du
       prompt et par entrée du vocabulaire ;
    3. la ligne utile pour générer est la dernière, de shape (1, 16), et l'argmax greedy en
       tire un id unique de shape (1, 1) ;
    4. le cache laissé derrière contient exactement la longueur du prompt, 3 positions.

    API à faire émerger (cible roadmap : `src/inference_lab/inference/prefill.py`)
    ----------------------------------------------------------------------------
        def prefill(
            model: torch.nn.Module,
            prompt_ids: torch.Tensor,
        ) -> tuple[torch.Tensor, KVCache]: ...

    Indice : `prefill` est un `model(prompt_ids, cache=cache)` unique, pas une boucle. Pour
    l'argmax greedy, `logits[:, -1, :].argmax(dim=-1, keepdim=True)` conserve la shape
    (1, 1) attendue en entrée du decode. Piège : `logits[:, -1]` sans `keepdim` te rend un
    tenseur de rang 1 et casse la concaténation sur l'axe séquence.
    """

    pytest.skip("Roadmap TDD 4.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.inference.prefill import prefill

    # Arrange — dimensions jouets constantes dans tout le fichier : batch=1, num_layers=2,
    #           num_heads=2, num_kv_heads=1, head_dim=4 (hidden=8), vocab=16, prompt de
    #           3 tokens. Construire `prompt_ids`, ids `torch.long` de shape (1, 3) et de
    #           valeurs DISTINCTES, et `model`, un modèle causal jouet déterministe (seed
    #           fixée) acceptant un KV cache et exposant un compteur `forward_calls`.

    # Act — appeler `prefill` sur le prompt complet et récupérer `logits` et `cache`, puis
    #       déduire `next_token_id`, l'id greedy tiré de la dernière ligne de logits.

    # Assert 1 — une seule passe de modèle pour tout le prompt
    assert model.forward_calls == 1

    # Assert 2 — un vecteur de logits par position du prompt
    assert logits.shape == (1, 3, 16)
    assert logits.dtype is torch.float32

    # Assert 3 — seule la dernière position sert à générer le token suivant
    assert logits[:, -1, :].shape == (1, 16)
    assert next_token_id.shape == (1, 1)
    assert next_token_id.dtype is torch.long

    # Assert 4 — le cache est rempli à la longueur du prompt
    assert cache.seq_len == 3


@pytest.mark.tdd
def test_prefill_populates_initial_kv_cache():
    """Roadmap 4.4 — le prefill laisse un cache prêt à l'emploi pour le decode.

    Objectif d'apprentissage
    ------------------------
    Le prefill n'a pas seulement pour but de produire un premier token : son vrai produit
    est l'ÉTAT, c'est-à-dire les K et V de toutes les couches pour toutes les positions du
    prompt. C'est cet état que le decode réutilise, et c'est lui qui occupe la mémoire du
    GPU en plus des poids.

    Sur ces dimensions jouets le cache pèse 192 octets ; la même formule appliquée à
    Qwen2.5-0.5B (24 couches, 2 têtes KV, head_dim 64) donne environ 12 Ko par token, soit
    plusieurs Go sur un long contexte. La section 5 généralisera ce calcul.

    Schéma mental
    -------------
        batch=1, num_layers=2, num_kv_heads=1, head_dim=4, prompt de 3 tokens, float32

        couche 0 : K (1, 1, 3, 4) = 12 éléments   V (1, 1, 3, 4) = 12 éléments
        couche 1 : K (1, 1, 3, 4) = 12 éléments   V (1, 1, 3, 4) = 12 éléments
                                                  --
        48 éléments x 4 octets = 192 octets, soit 192 / 3 = 64 octets par token

    Ce que ce test vérifie
    ----------------------
    1. le cache contient une paire (K, V) pour chacune des 2 couches, à la longueur du
       prompt (3 positions) ;
    2. chaque tenseur stocké a la shape (1, 1, 3, 4), et K n'est pas V ;
    3. les 3 positions mémorisées sont distinctes les unes des autres : le prompt a bien
       été encodé position par position, pas dupliqué ;
    4. la mémoire du cache est de 192 octets en float32, soit 64 octets par token.

    API à faire émerger (cible roadmap : `src/inference_lab/inference/prefill.py`)
    ----------------------------------------------------------------------------
        logits, cache = prefill(model, prompt_ids)
        cache.get(layer_idx) -> tuple[torch.Tensor, torch.Tensor]

    Indice : la mémoire se mesure avec `tenseur.numel() * tenseur.element_size()`, la
    formule de la section 1.9. Piège : n'alloue pas le cache à une longueur maximale
    remplie de zéros pour cet exercice — la longueur doit être exactement celle du prompt,
    sinon les asserts de shape et d'octets tombent.
    """

    pytest.skip("Roadmap TDD 4.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.inference.prefill import prefill

    # Arrange — mêmes dimensions jouets : batch=1, num_layers=2, num_kv_heads=1, head_dim=4,
    #           vocab=16, prompt de 3 tokens aux ids DISTINCTS (`prompt_ids`, shape (1, 3),
    #           `torch.long`), et `model`, le même modèle causal jouet déterministe en
    #           float32 acceptant un KV cache.

    # Act — appeler `prefill` pour obtenir `cache`, relire les paires (K, V) des deux couches,
    #       et calculer `cache_bytes`, la somme des `numel() * element_size()` de tous les
    #       tenseurs stockés.

    # Assert 1 — une paire (K, V) par couche, à la longueur du prompt
    assert len(cache) == 2
    assert cache.seq_len == 3

    # Assert 2 — shapes (batch, num_kv_heads, seq, head_dim) et K distinct de V
    assert cache.get(0)[0].shape == (1, 1, 3, 4)
    assert cache.get(0)[1].shape == (1, 1, 3, 4)
    assert cache.get(1)[0].shape == (1, 1, 3, 4)
    assert not torch.equal(cache.get(0)[0], cache.get(0)[1])

    # Assert 3 — les 3 positions du prompt ont chacune leur propre K
    assert not torch.equal(cache.get(0)[0][:, :, 0, :], cache.get(0)[0][:, :, 1, :])
    assert not torch.equal(cache.get(0)[0][:, :, 1, :], cache.get(0)[0][:, :, 2, :])

    # Assert 4 — mémoire du cache, chiffrée : 48 éléments float32
    assert cache.get(0)[0].numel() == 12
    assert cache_bytes == 192
    assert cache_bytes // 3 == 64
