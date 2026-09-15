"""Section 4.2 / 4.3 — structure et croissance du KV cache.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/04_inference/test_kv_cache.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(2 couches, shapes (1, 1, 3, 4), longueurs 3 puis 4 puis 5) : c'est volontaire. Un test
doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Cette section pose la STRUCTURE de données qui supprime le gâchis mesuré en 4.1 : une
paire (K, V) par couche du modèle, concaténée le long de l'axe séquence. Tout le reste de
la roadmap s'appuie sur ce contrat : prefill (4.4) le remplit d'un coup, decode (4.5) y
ajoute une position par étape, et la section 5 en calcule la mémoire.

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
def test_kv_cache_stores_keys_and_values_for_every_layer():
    """Roadmap 4.2 — le cache est un tableau de paires (K, V), une par couche.

    Objectif d'apprentissage
    ------------------------
    Un transformer recalcule l'attention à CHAQUE couche : les clés de la couche 0 ne
    servent à rien pour la couche 1, car chaque couche a ses propres projections W_k et
    W_v. Le cache doit donc être indexé par couche, et pour chaque couche stocker deux
    tenseurs de shape (batch, num_kv_heads, seq, head_dim).

    Retenir cette shape est essentiel : c'est elle qui donne la formule mémoire de la
    section 5 (`2 x num_layers x num_kv_heads x head_dim x seq x octets_par_élément`) et
    c'est l'axe `num_kv_heads` que GQA réduit en 4.8.

    Schéma mental
    -------------
        batch=1, num_layers=2, num_kv_heads=1, head_dim=4, prompt de 3 tokens

        cache
          |-- couche 0 : K (1, 1, 3, 4)   V (1, 1, 3, 4)
          |-- couche 1 : K (1, 1, 3, 4)   V (1, 1, 3, 4)

        axe 2 = séquence : c'est le SEUL axe qui grandit pendant la génération

    Ce que ce test vérifie
    ----------------------
    1. le cache expose une entrée par couche du modèle (2 couches) et chaque entrée est
       une paire (K, V) ;
    2. chaque tenseur stocké a la shape (batch=1, num_kv_heads=1, seq=3, head_dim=4) ;
    3. les couches sont indépendantes : écrire dans la couche 1 ne modifie pas la couche 0 ;
    4. la longueur courante du cache est celle du prompt, soit 3 positions, en float32.

    API à faire émerger (cible roadmap : `src/inference_lab/cache/kv_cache.py`)
    -------------------------------------------------------------------------
        class KVCache:
            def __init__(self, num_layers: int) -> None: ...
            def append(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor)
                -> None: ...
            def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...
            def __len__(self) -> int: ...          # nombre de couches
            @property
            def seq_len(self) -> int: ...          # positions déjà mémorisées

        La cible est un chemin concret de la roadmap : c'est bien ce module qu'il faut créer.

    Indice : `torch.cat([ancien, nouveau], dim=2)` suffit pour accumuler, et le cas
    « couche encore vide » se traite en stockant directement le premier tenseur. Piège :
    n'utilise pas `dim=-2` par réflexe sans vérifier le rang, et ne mets pas une seule
    paire partagée entre les couches — une liste de longueur `num_layers`.
    """

    pytest.skip("Roadmap TDD 4.2 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.cache.kv_cache import KVCache

    # Arrange — dimensions jouets constantes dans tout le fichier : batch=1, num_layers=2,
    #           num_heads=2, num_kv_heads=1, head_dim=4, prompt de 3 tokens. Construire
    #           `cache`, un `KVCache` de 2 couches, puis quatre tenseurs `torch.float32` de
    #           shape (1, 1, 3, 4) : `layer0_keys`, `layer0_values`, `layer1_keys`,
    #           `layer1_values`. Les valeurs doivent DIFFÉRER d'une couche à l'autre (seed
    #           fixée ou tenseurs construits distincts), sinon l'assert 3 ne prouve rien.

    # Act — enregistrer dans le cache la paire (K, V) de la couche 0 puis celle de la
    #       couche 1, et relire les deux couches.

    # Assert 1 — une entrée par couche, chaque entrée est une paire (K, V)
    assert isinstance(cache, KVCache)
    assert len(cache) == 2
    assert len(cache.get(0)) == 2
    assert len(cache.get(1)) == 2

    # Assert 2 — shape attendue : (batch, num_kv_heads, seq, head_dim)
    assert cache.get(0)[0].shape == (1, 1, 3, 4)
    assert cache.get(0)[1].shape == (1, 1, 3, 4)
    assert cache.get(1)[0].shape == (1, 1, 3, 4)
    assert cache.get(1)[1].shape == (1, 1, 3, 4)

    # Assert 3 — les couches ne se marchent pas dessus
    assert torch.equal(cache.get(0)[0], layer0_keys)
    assert torch.equal(cache.get(1)[0], layer1_keys)
    assert not torch.equal(cache.get(0)[0], cache.get(1)[0])

    # Assert 4 — la longueur du cache est le nombre de positions mémorisées
    assert cache.seq_len == 3
    assert cache.get(0)[0].dtype is torch.float32


@pytest.mark.tdd
def test_kv_cache_grows_one_position_per_decode_step():
    """Roadmap 4.3 — une étape de decode ajoute exactement une position au cache.

    Objectif d'apprentissage
    ------------------------
    Le cache est la mémoire de la génération : il ne grandit que d'un token par étape, et
    seulement sur l'axe séquence. C'est ce qui rend le coût du decode constant par token
    côté calcul, mais croissant côté mémoire — la vraie limite d'un moteur d'inférence
    (le cache finit par dépasser la taille des poids sur les longs contextes).

    Deux erreurs classiques sont interdites par ce test : ajouter plus d'une position par
    étape (on a réencodé tout le contexte, cf. 4.1), et écraser l'historique au lieu de le
    concaténer (le modèle perdrait la mémoire du prompt).

    Schéma mental
    -------------
        batch=1, num_layers=2, num_kv_heads=1, head_dim=4

        après prefill  : K (1, 1, 3, 4)   <- 3 tokens de prompt
        + decode t3    : K (1, 1, 4, 4)   <- +1 position
        + decode t4    : K (1, 1, 5, 4)   <- +1 position

        longueurs : 3 -> 4 -> 5, et les positions 0..2 restent bit à bit identiques

    Ce que ce test vérifie
    ----------------------
    1. après le prompt, le cache contient 3 positions ;
    2. chaque étape de decode ajoute exactement une position : 3 -> 4 -> 5 ;
    3. toutes les couches grandissent ensemble, aucune ne prend du retard ;
    4. l'historique est concaténé et non réécrit : les 3 premières positions sont
       inchangées et la position 3 contient exactement le K de la première étape.

    API à faire émerger (cible roadmap : `src/inference_lab/cache/kv_cache.py`)
    -------------------------------------------------------------------------
        cache.append(layer_idx, keys, values)   # keys/values de shape (1, 1, 1, 4)
        cache.seq_len -> int

    Indice : `seq_len` se lit sur la couche 0 (`self._keys[0].shape[2]`) puisque toutes les
    couches sont synchronisées. Piège : `torch.stack` ajoute un axe, ce n'est pas ce qu'on
    veut ; c'est `torch.cat(..., dim=2)` qui allonge l'axe séquence existant.
    """

    pytest.skip("Roadmap TDD 4.3 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.cache.kv_cache import KVCache

    # Arrange — mêmes dimensions jouets : batch=1, num_layers=2, num_kv_heads=1, head_dim=4,
    #           prompt de 3 tokens. Construire `cache` (2 couches), les tenseurs de prompt
    #           `prompt_keys` / `prompt_values` de shape (1, 1, 3, 4), puis deux paires de
    #           decode `step1_keys` / `step1_values` et `step2_keys` / `step2_values`, de
    #           shape (1, 1, 1, 4) chacune, aux valeurs distinctes du prompt (seed fixée).

    # Act — remplir les deux couches avec le prompt, puis jouer deux étapes de decode en
    #       ajoutant à chaque étape une seule position à CHACUNE des deux couches, en relevant
    #       `cache.seq_len` dans `seq_len_after_prompt`, `seq_len_after_step1` et
    #       `seq_len_after_step2`.

    # Assert 1 — le prompt occupe 3 positions, une par token
    assert prompt_keys.shape == (1, 1, 3, 4)
    assert seq_len_after_prompt == 3

    # Assert 2 — une position par étape de decode : 3 puis 4 puis 5
    assert step1_keys.shape == (1, 1, 1, 4)
    assert seq_len_after_step1 == 4
    assert seq_len_after_step2 == 5
    assert seq_len_after_step2 - seq_len_after_step1 == 1

    # Assert 3 — les deux couches restent synchronisées
    assert cache.get(0)[0].shape == (1, 1, 5, 4)
    assert cache.get(1)[0].shape == (1, 1, 5, 4)
    assert cache.get(1)[1].shape == (1, 1, 5, 4)

    # Assert 4 — l'historique est concaténé, jamais réécrit
    assert torch.equal(cache.get(0)[0][:, :, :3, :], prompt_keys)
    assert torch.equal(cache.get(0)[0][:, :, 3:4, :], step1_keys)
    assert torch.equal(cache.get(0)[1][:, :, 4:5, :], step2_values)
