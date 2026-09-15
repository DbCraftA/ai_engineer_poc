"""Section 2.8 / 2.9 — GQA et MQA : partager les têtes K/V pour réduire le KV cache.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/02_attention/test_grouped_query_attention.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(2 têtes de query, 1 tête KV, 16 éléments contre 32, facteur 2) : c'est volontaire. Un test
doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Dimensions constantes de la section : batch=1, seq=4, hidden=8, num_heads=2, head_dim=4 et
num_kv_heads=1. Avec ce jeu de dimensions, num_kv_heads=1 est à la fois le cas GQA minimal
(2 têtes de query pour 1 tête KV) et le cas MQA (une seule tête KV) : les deux notions se
rejoignent, et c'est précisément ce que le test 2.9 rend explicite. Enjeu réel : le KV cache
est le poste mémoire dominant en decode, sa taille est divisée par num_heads / num_kv_heads
(Qwen2.5-0.5B : 14 têtes de query pour 2 têtes KV, soit 7 fois moins de cache).

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
def test_gqa_uses_fewer_kv_heads_than_query_heads():
    """Roadmap 2.8 — Q garde toutes ses têtes, K et V en ont moins.

    Objectif d'apprentissage
    ------------------------
    En decode, le modèle relit à chaque token TOUT le KV cache : le débit est limité par la
    bande passante mémoire, pas par le calcul. GQA attaque directement ce coût en réduisant
    le nombre de têtes de K et V, sans toucher au nombre de têtes de query : la capacité
    d'attention reste répartie sur `num_heads` sous-espaces, mais on ne stocke plus que
    `num_kv_heads` paires (K, V) par token et par couche. La sortie garde la shape d'une
    attention multi-têtes classique, ce qui rend GQA remplaçable sans changer le reste du bloc.

    Schéma mental
    -------------
        Q  (batch=1, num_heads=2,    seq=4, head_dim=4)   -> 32 éléments
        K  (1,        num_kv_heads=1, seq=4, head_dim=4)   -> 16 éléments
        V  (1,        num_kv_heads=1, seq=4, head_dim=4)   -> 16 éléments

        MHA équivalent : K et V en (1, 2, 4, 4) -> 32 éléments chacun
        gain KV cache = num_heads / num_kv_heads = 2 / 1 = 2

        sortie = (1, 2, 4, 4) : autant de têtes que Q, comme en MHA

    Ce que ce test vérifie
    ----------------------
    1. K et V ont strictement moins de têtes que Q, et la sortie garde les 2 têtes de Q ;
    2. le KV cache est deux fois plus petit qu'en MHA (16 éléments contre 32) ;
    3. la sortie reste finie et de même dtype : GQA est un remplacement direct de MHA ;
    4. un nombre de têtes de query non divisible par le nombre de têtes KV est refusé.

    API à faire émerger (cible roadmap : `src/inference_lab/nn/attention/gqa.py`)
    ---------------------------------------------------------------------------
        def grouped_query_attention(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
        ) -> torch.Tensor: ...

    Indice : la fonction peut réutiliser `nn/attention/naive.py` (scores, softmax, sortie)
    après avoir aligné les têtes de K et V sur celles de Q. Piège : ne JAMAIS matérialiser le
    K répété dans le cache, sinon le gain mémoire disparaît — la répétition est faite au
    moment du calcul. Lève une `ValueError` si `num_heads % num_kv_heads != 0`.
    """

    pytest.skip("Roadmap TDD 2.8 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.attention.gqa import grouped_query_attention

    # Arrange — batch=1, seq=4, head_dim=4, num_heads=2, num_kv_heads=1.
    #           `q` : tenseur (1, 2, 4, 4) en `torch.float32`, déterministe (seed fixée).
    #           `k`, `v` : deux tenseurs (1, 1, 4, 4) en `torch.float32`, distincts.
    #           `k_mha` : un tenseur (1, 2, 4, 4), ce que coûterait le même cache en MHA.
    #           `k_bad` : un tenseur (1, 3, 4, 4), configuration incohérente avec 2 têtes de
    #           query (2 n'est pas divisible par 3).

    # Act — calculer `out = grouped_query_attention(q, k, v)`.

    # Assert 1 — moins de têtes KV que de têtes de query, mais autant de têtes en sortie
    assert q.shape[1] == 2
    assert k.shape[1] == 1
    assert v.shape[1] == 1
    assert out.shape == (1, 2, 4, 4)

    # Assert 2 — le KV cache est divisé par num_heads / num_kv_heads = 2
    assert k.numel() == 16
    assert k_mha.numel() == 32
    assert k.numel() * 2 == k_mha.numel()

    # Assert 3 — remplacement direct de MHA : même dtype, aucune valeur non finie
    assert out.dtype is torch.float32
    assert bool(out.isfinite().all())

    # Assert 4 — 2 têtes de query pour 3 têtes KV : configuration impossible
    with pytest.raises(ValueError):
        grouped_query_attention(q, k_bad, k_bad)


@pytest.mark.tdd
def test_multiple_query_heads_share_key_value_heads():
    """Roadmap 2.8 — chaque tête KV est répétée à l'identique pour son groupe de queries.

    Objectif d'apprentissage
    ------------------------
    Le mécanisme de partage tient en une opération : chaque tête KV est dupliquée pour couvrir
    les têtes de query de son groupe. La duplication doit être CONTIGUË par groupe
    (`repeat_interleave`), pas cyclique (`repeat`) : les têtes 0 et 1 lisent la tête KV 0, les
    têtes 2 et 3 la tête KV 1. Se tromper d'opération donne des shapes correctes et un modèle
    qui apprend n'importe quoi — c'est le bug classique au chargement des poids d'un modèle
    GQA comme Qwen2.5. Une fois la répétition faite, GQA est exactement une MHA.

    Schéma mental
    -------------
        cas de la section (num_kv_heads=1, num_heads=2) :
            kv (1, 1, 4, 4)  --repeat_interleave(2, dim=1)-->  (1, 2, 4, 4)
            les 2 emplacements de tête contiennent la MÊME donnée

        cas d'ordre (num_kv_heads=2, num_heads=4), pour distinguer les deux opérations :
            têtes KV       :  [A, B]
            repeat_interleave -> [A, A, B, B]   (attendu)
            repeat            -> [A, B, A, B]   (faux)

    Ce que ce test vérifie
    ----------------------
    1. la répétition rend le nombre de têtes de query, ici (1, 2, 4, 4) ;
    2. les emplacements de tête d'un même groupe contiennent des copies identiques de la
       tête KV d'origine ;
    3. l'ordre est bien groupé ([A, A, B, B]) et non cyclique ([A, B, A, B]) ;
    4. GQA appliquée à des K/V déjà répétés donne le même résultat qu'à des K/V partagés :
       GQA est une MHA dont on a factorisé le cache.

    API à faire émerger (cible roadmap : `src/inference_lab/nn/attention/gqa.py`)
    ---------------------------------------------------------------------------
        def repeat_kv_heads(kv: torch.Tensor, num_heads: int) -> torch.Tensor: ...

    Indice : `kv.repeat_interleave(num_heads // num_kv_heads, dim=1)`, ou la version sans
    copie `kv[:, :, None].expand(...)` suivie d'un `reshape`. Piège : `kv.repeat(1, n, 1, 1)`
    a la bonne shape mais l'ordre cyclique — l'assert 3 est là pour ça.
    """

    pytest.skip("Roadmap TDD 2.8 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.attention.gqa import grouped_query_attention, repeat_kv_heads

    # Arrange — batch=1, seq=4, head_dim=4, num_heads=2, num_kv_heads=1.
    #           `q` : tenseur (1, 2, 4, 4) en `torch.float32`, déterministe (seed fixée).
    #           `k`, `v` : deux tenseurs (1, 1, 4, 4), valeurs toutes distinctes.
    #           `kv_two` : un tenseur (1, 2, 4, 4) servant au seul cas d'ordre, dont les deux
    #           têtes ont des contenus clairement différents.

    # Act — calculer `repeated = repeat_kv_heads(k, 2)` et `repeated_two` en répétant `kv_two`
    #       sur 4 têtes ; calculer `out_shared = grouped_query_attention(q, k, v)` puis
    #       `out_repeated` en appelant `grouped_query_attention` sur `q` et les versions déjà
    #       répétées de `k` et `v`.

    # Assert 1 — après répétition, autant de têtes KV que de têtes de query
    assert repeated.shape == (1, 2, 4, 4)
    assert repeat_kv_heads(k, 2).shape == (1, 2, 4, 4)

    # Assert 2 — les deux emplacements sont des copies de la même tête KV
    assert torch.equal(repeated[0, 0], repeated[0, 1])
    assert torch.equal(repeated[0, 0], k[0, 0])

    # Assert 3 — ordre groupé [A, A, B, B], pas cyclique [A, B, A, B]
    assert repeated_two.shape == (1, 4, 4, 4)
    assert torch.equal(repeated_two[0, 0], kv_two[0, 0])
    assert torch.equal(repeated_two[0, 1], kv_two[0, 0])
    assert torch.equal(repeated_two[0, 2], kv_two[0, 1])
    assert not torch.equal(repeated_two[0, 1], kv_two[0, 1])

    # Assert 4 — GQA sur K/V partagés == MHA sur K/V répétés
    torch.testing.assert_close(out_shared, out_repeated, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        out_shared,
        grouped_query_attention(q, repeat_kv_heads(k, 2), repeat_kv_heads(v, 2)),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.tdd
def test_mqa_uses_single_key_value_head():
    """Roadmap 2.9 — MQA est le cas extrême de GQA : une seule tête K/V pour tout le monde.

    Objectif d'apprentissage
    ------------------------
    MQA pousse le partage à son maximum : `num_kv_heads = 1`. Le KV cache est alors divisé par
    `num_heads` (facteur 14 sur Qwen2.5-0.5B), ce qui autorise des batchs et des contextes bien
    plus grands à mémoire constante. Le prix est une perte de diversité : toutes les têtes de
    query lisent exactement les mêmes clés et les mêmes valeurs, seules leurs requêtes
    diffèrent. GQA avec 2 à 8 têtes KV est le compromis retenu en pratique ; ici, avec
    num_heads=2, GQA à 1 tête KV et MQA coïncident, ce qui rend le continuum visible :
    MHA (num_kv_heads = num_heads) ... GQA ... MQA (num_kv_heads = 1).

    Schéma mental
    -------------
        MHA : Q 2 têtes, K/V 2 têtes  -> cache 32 éléments par tenseur
        MQA : Q 2 têtes, K/V 1 tête   -> cache 16 éléments par tenseur, gain 2

        toutes les têtes de query lisent la même tête KV :
            repeat_kv_heads(k_mqa, 2)[0, 0] == repeat_kv_heads(k_mqa, 2)[0, 1] == k_mqa[0, 0]

        Qwen2.5-0.5B (14 têtes de query) en MQA : cache divisé par 14

    Ce que ce test vérifie
    ----------------------
    1. K et V n'ont qu'UNE tête, et la sortie garde les 2 têtes de query ;
    2. toutes les têtes de query lisent des clés strictement identiques ;
    3. le cache est divisé par num_heads = 2 par rapport à MHA (16 éléments contre 32) ;
    4. MQA est bien le cas limite de GQA : passer par la répétition explicite donne le même
       résultat numérique.

    API à faire émerger (cible roadmap : `src/inference_lab/nn/attention/gqa.py`)
    ---------------------------------------------------------------------------
        def grouped_query_attention(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
        ) -> torch.Tensor: ...
        def repeat_kv_heads(kv: torch.Tensor, num_heads: int) -> torch.Tensor: ...

    Indice : aucun code spécifique à MQA n'est nécessaire — si `grouped_query_attention` gère
    un groupe de taille `num_heads`, MQA fonctionne déjà. Piège : oublier la dimension de tête
    et passer un K de shape (1, 4, 4) ; garde toujours 4 dimensions, y compris quand la
    dimension de tête vaut 1.
    """

    pytest.skip("Roadmap TDD 2.9 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.nn.attention.gqa import grouped_query_attention, repeat_kv_heads

    # Arrange — batch=1, seq=4, head_dim=4, num_heads=2, num_kv_heads=1.
    #           `q` : tenseur (1, 2, 4, 4) en `torch.float32`, déterministe (seed fixée), dont
    #           les deux têtes ont des contenus différents.
    #           `k_mqa`, `v_mqa` : deux tenseurs (1, 1, 4, 4), l'unique tête K/V.
    #           `k_mha` : un tenseur (1, 2, 4, 4), le cache qu'exigerait une MHA.

    # Act — calculer `out = grouped_query_attention(q, k_mqa, v_mqa)`, `keys = repeat_kv_heads(
    #       k_mqa, 2)`, et `out_explicit` en appelant `grouped_query_attention` sur `q` et les
    #       versions répétées de `k_mqa` et `v_mqa`.

    # Assert 1 — une seule tête K/V, mais toujours 2 têtes en sortie
    assert k_mqa.shape == (1, 1, 4, 4)
    assert v_mqa.shape == (1, 1, 4, 4)
    assert out.shape == (1, 2, 4, 4)
    assert grouped_query_attention(q, k_mqa, v_mqa).shape == (1, 2, 4, 4)

    # Assert 2 — les deux têtes de query lisent exactement les mêmes clés
    assert torch.equal(keys[0, 0], keys[0, 1])
    assert torch.equal(keys[0, 0], k_mqa[0, 0])
    assert torch.equal(repeat_kv_heads(k_mqa, 2), keys)

    # Assert 3 — cache divisé par num_heads = 2 : 16 éléments au lieu de 32
    assert k_mqa.numel() == 16
    assert k_mqa.numel() * 2 == k_mha.numel()

    # Assert 4 — MQA = GQA avec un groupe de taille num_heads
    torch.testing.assert_close(out, out_explicit, atol=1e-6, rtol=1e-6)
