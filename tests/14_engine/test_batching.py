"""Section 14.1 — static batching : grouper les requêtes, et payer le padding.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/14_engine/test_batching.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(batch (3, 8), 24 positions, 8 de padding, 10 étapes de decode) : c'est volontaire. Un test
doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Ce chapitre est le moteur d'inférence, inspiré de vLLM. On y quitte les tenseurs pour de la
LOGIQUE pure : ordonnancement, allocation de blocs, tables de correspondance. Tout est
déterministe, sans GPU, avec des nombres exacts. Le static batching est le point de départ
naïf : on attend N requêtes, on les exécute ensemble, et on découvre deux gâchis chiffrés —
le padding au prefill, et l'attente de la requête la plus lente au decode. Les sections 14.2
et 14.3 les suppriment.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices. Même raison pour l'import
# du module cible : c'est ton code d'`Act` qui l'appellera, le linter le voit donc inutilisé.
# ruff: noqa: F401, F821

import pytest


@pytest.mark.tdd
def test_static_batch_groups_requests_before_execution():
    """Roadmap 14.1 — un batch statique est rectangulaire, donc padé, donc gaspilleur.

    Objectif d'apprentissage
    ------------------------
    Un GPU n'aime qu'une chose : les tenseurs rectangulaires. Pour traiter trois prompts de
    longueurs différentes en un seul appel, il faut donc les aligner sur la longueur MAXIMALE
    et remplir le reste avec un token de padding masqué. Ce padding est calculé pour rien :
    il consomme des FLOPs, de la bande passante et des slots de KV cache sans produire aucun
    token utile.

    Le second gâchis est temporel. Un batch statique est figé de son ouverture à sa
    fermeture : les requêtes courtes restent dans le batch jusqu'à ce que la plus longue ait
    fini de générer, et leurs slots continuent d'être calculés à vide. Sur un serveur réel,
    cela se lit directement sur les métriques de la section 13 : le TTFT des requêtes qui
    attendent le remplissage du batch explose, et le débit s'effondre dès que les longueurs
    sont hétérogènes.

    Schéma mental
    -------------
        3 prompts de 3, 5 et 8 tokens, `P` = padding

            r0 : t t t P P P P P
            r1 : t t t t t P P P
            r2 : t t t t t t t t

        batch (3, 8) = 24 positions calculées, 16 vraies, 8 de padding -> 33 % perdus

        decode, longueurs de sortie 2, 6 et 10 tokens :

            étapes  : 10 (celle de la requête la plus lente)
            slots   : 3 x 10 = 30, dont 18 utiles et 12 calculés à vide -> 40 % perdus

    Ce que ce test vérifie
    ----------------------
    1. le batch est un tenseur rectangulaire (3, 8) padé à la longueur du plus long prompt,
       et son masque marque exactement 3, 5 et 8 vrais tokens ;
    2. le coût du padding en positions exactes : 24 positions pour 16 vrais tokens, donc 8
       positions de padding, soit un tiers du calcul de prefill jeté ;
    3. le batch avance au rythme du plus lent : 10 étapes de decode, 30 slots de séquence
       calculés pour 18 tokens utiles, donc 12 slots à vide ;
    4. le gâchis vient de l'HÉTÉROGÉNÉITÉ, pas du batching : à longueurs égales, le padding
       tombe à zéro sans changer la taille du batch.

    API à faire émerger (cible roadmap : `src/inference_lab/engine/batching.py`)
    -------------------------------------------------------------------------
        def build_static_batch(
            prompts: Sequence[Sequence[int]], pad_id: int
        ) -> tuple[torch.Tensor, torch.Tensor]: ...      # (tokens, attention_mask)

        @dataclass(frozen=True)
        class PaddingStats:
            batch_shape: tuple[int, int]
            capacity_positions: int
            real_tokens: int
            padding_positions: int
            padding_ratio: float

        def static_batch_padding_stats(prompt_lengths: Sequence[int]) -> PaddingStats: ...

        @dataclass(frozen=True)
        class DecodeStats:
            num_steps: int
            capacity_slots: int
            useful_slots: int
            wasted_slots: int

        def static_batch_decode_stats(output_lengths: Sequence[int]) -> DecodeStats: ...

        La cible est un chemin concret de la roadmap : c'est bien ce module qu'il faut créer.

    Indice : tout tient en `max(...)`, `sum(...)` et une soustraction, aucune bibliothèque
    n'est nécessaire au-delà de `torch.full` pour le tenseur padé. Piège classique : padder à
    droite ET construire le masque dans la même passe, sinon les deux divergent. Second
    piège : `padding_ratio` se rapporte à la CAPACITÉ du batch (24), pas au nombre de vrais
    tokens (16) — sinon on obtient 0,5 au lieu d'un tiers.
    """

    pytest.skip("Roadmap TDD 14.1 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.engine.batching import (
        build_static_batch,
        static_batch_decode_stats,
        static_batch_padding_stats,
    )

    # Arrange — trois requêtes de longueurs volontairement inégales : `prompts`, une liste de
    #           trois listes d'identifiants de tokens de 3, 5 et 8 éléments (valeurs
    #           quelconques mais toutes différentes de `pad_id`), `prompt_lengths` la liste
    #           [3, 5, 8] de leurs longueurs, `pad_id` l'identifiant réservé au padding, et
    #           `output_lengths` les longueurs de génération attendues, 2, 6 et 10 tokens.
    #           Aucun modèle, aucun GPU : cette section ne manipule que des entiers.

    # Act — construire le batch statique padé (`tokens` et `attention_mask`), puis demander à
    #       l'API les statistiques de padding (`pad_stats`) et de decode (`decode_stats`).

    # Assert 1 — le batch est rectangulaire, padé à la longueur du plus long prompt
    assert tokens.shape == (3, 8)
    assert attention_mask.shape == (3, 8)
    assert attention_mask.sum(dim=1).tolist() == [3, 5, 8]
    assert attention_mask.sum().item() == 16
    assert (tokens[0, 3:] == pad_id).all()

    # Assert 2 — le coût du padding au prefill, en positions exactes
    assert pad_stats.batch_shape == (3, 8)
    assert pad_stats.capacity_positions == 24
    assert pad_stats.real_tokens == 16
    assert pad_stats.padding_positions == 8
    assert pad_stats.padding_ratio == pytest.approx(1 / 3, rel=1e-12)

    # Assert 3 — au decode, le batch entier attend la requête la plus lente
    assert decode_stats.num_steps == 10
    assert decode_stats.capacity_slots == 30
    assert decode_stats.useful_slots == 18
    assert decode_stats.wasted_slots == 12
    assert decode_stats.wasted_slots / decode_stats.capacity_slots == 0.4

    # Assert 4 — le gâchis vient de l'hétérogénéité des longueurs, pas du batch lui-même
    assert static_batch_padding_stats([8, 8, 8]).capacity_positions == 24
    assert static_batch_padding_stats([8, 8, 8]).padding_positions == 0
    assert static_batch_padding_stats([8, 8, 8]).padding_ratio == 0.0
