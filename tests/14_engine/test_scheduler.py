"""Section 14.2 / 14.3 — ordonnanceur : continuous batching et admission sous contrainte.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/14_engine/test_scheduler.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(5 étapes au lieu de 7, 2 séquences actives et 2 en attente, 3 blocs libres sur 8) : c'est
volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même formule
que le code testé.

C'est le cœur du moteur. 14.2 supprime le gâchis temporel mesuré en 14.1 : le batch devient
mutable, une séquence terminée libère sa place immédiatement. 14.3 ajoute la contrainte du
monde réel : la mémoire est finie (section 5), donc au-delà d'une certaine charge le
scheduler met en attente au lieu d'accepter. Les blocs de KV cache comptés ici sont ceux des
sections 14.4 à 14.6. Tests de logique pure : aucun tenseur, aucun GPU.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices. Même raison pour l'import
# du module cible : c'est ton code d'`Act` qui l'appellera, le linter le voit donc inutilisé.
# ruff: noqa: F401, F821

import pytest


@pytest.mark.tdd
def test_finished_sequences_can_leave_batch_while_others_continue():
    """Roadmap 14.2 — continuous batching : le batch se recompose à chaque étape.

    Objectif d'apprentissage
    ------------------------
    En static batching (14.1), le batch est un bloc figé : il s'ouvre, s'exécute, se ferme.
    En continuous batching, le batch est recomposé AVANT chaque étape de decode. Dès qu'une
    séquence émet son token de fin, elle quitte le batch, rend ses blocs de KV cache, et la
    première requête en attente prend sa place à l'étape suivante — sans attendre que les
    autres aient terminé.

    Deux conséquences directes sur les métriques de la section 13 : le TTFT des requêtes
    tardives s'effondre (elles n'attendent plus la fin du batch courant, seulement la fin
    d'UNE séquence), et le débit monte parce qu'aucun slot du batch n'est calculé à vide.
    C'est la technique qui a fait la réputation de vLLM et de TGI, et elle ne demande aucune
    modification du modèle : uniquement de la logique d'ordonnancement.

    Schéma mental
    -------------
        capacité : 2 séquences simultanées
        requêtes : A (2 tokens à générer), B (4 tokens), C (3 tokens), toutes déjà arrivées

            étape 1 : [A, B]      A finit à la fin de l'étape 2
            étape 2 : [A, B]      A sort, sa place se libère immédiatement
            étape 3 : [B, C]      C entre alors que B est encore en cours
            étape 4 : [B, C]      B sort
            étape 5 : [C]

        total : 5 étapes et 9 slots exécutés pour 9 tokens utiles (aucun slot à vide)
        static batching à capacité 2 : (A, B) pendant 4 étapes puis (C) pendant 3 = 7 étapes

    Ce que ce test vérifie
    ----------------------
    1. la composition du batch étape par étape, en identifiants de requêtes :
       [A, B], [A, B], [B, C], [B, C], [C], soit 5 étapes ;
    2. C entre dès que A libère sa place, alors que B est toujours en cours d'exécution, et
       l'ordre d'achèvement est A puis B puis C ;
    3. aucun slot n'est calculé à vide : 9 slots exécutés pour exactement 9 tokens générés ;
    4. le gain face au static batching de 14.1 à capacité identique : 5 étapes au lieu de 7.

    API à faire émerger (cible proposée : `src/inference_lab/engine/scheduler.py`)
    ----------------------------------------------------------------------------
        class ContinuousBatchScheduler:
            def __init__(self, max_num_seqs: int) -> None: ...
            def add_request(self, request_id: str, num_output_tokens: int) -> None: ...
            def step(self) -> list[str]: ...     # ids exécutés pendant CETTE étape
            @property
            def running_ids(self) -> list[str]: ...
            @property
            def waiting_ids(self) -> list[str]: ...
            @property
            def finished_ids(self) -> list[str]: ...   # dans l'ordre d'achèvement
            @property
            def has_work(self) -> bool: ...

        La roadmap indique seulement « scheduler » : on propose ce chemin concret, cohérent
        avec le package `engine` existant, et partagé avec les tests 14.3 et 14.8.

    Indice : l'ordre des opérations dans `step()` est TOUTE la difficulté — d'abord admettre
    les requêtes en attente tant qu'il reste de la place, puis exécuter le batch obtenu, puis
    seulement retirer les séquences qui viennent de terminer. Retirer avant d'admettre ferait
    entrer une requête une étape trop tôt et casserait l'assert 1. `num_output_tokens` est
    une simplification de l'exercice : dans un vrai moteur, c'est le token de fin émis par le
    modèle qui déclenche la sortie du batch.
    """

    pytest.skip("Roadmap TDD 14.2 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.engine.batching import static_batch_decode_stats
    from inference_lab.engine.scheduler import ContinuousBatchScheduler

    # Arrange — un `scheduler` de type `ContinuousBatchScheduler` limité à 2 séquences
    #           simultanées, et trois requêtes ajoutées dans cet ordre, identifiées par les
    #           chaînes "A", "B" et "C", devant générer respectivement 2, 4 et 3 tokens.
    #           Toutes sont présentes dès le départ : ce test isole la recomposition du
    #           batch, pas les arrivées échelonnées.

    # Act — faire tourner le scheduler étape par étape jusqu'à épuisement du travail, en
    #       collectant dans `batches` la liste retournée par chaque `step()`, c'est-à-dire la
    #       composition du batch à chaque étape.

    # Assert 1 — la composition du batch, étape par étape
    assert batches == [["A", "B"], ["A", "B"], ["B", "C"], ["B", "C"], ["C"]]
    assert len(batches) == 5

    # Assert 2 — une place libérée est réutilisée aussitôt, sans attendre les autres
    assert "C" not in batches[1]
    assert "A" not in batches[2]
    assert "B" in batches[2] and "B" in batches[3]
    assert scheduler.finished_ids == ["A", "B", "C"]
    assert scheduler.running_ids == [] and scheduler.waiting_ids == []
    assert scheduler.has_work is False

    # Assert 3 — aucun slot du batch n'est calculé à vide
    assert sum(len(batch) for batch in batches) == 9
    assert all(1 <= len(batch) <= 2 for batch in batches)

    # Assert 4 — le static batching de 14.1, à capacité 2, aurait demandé 7 étapes
    assert static_batch_decode_stats([2, 4]).num_steps == 4
    assert static_batch_decode_stats([3]).num_steps == 3
    assert static_batch_decode_stats([2, 4]).wasted_slots == 2
    assert len(batches) == 5


@pytest.mark.tdd
def test_scheduler_rejects_or_delays_requests_when_capacity_is_exhausted():
    """Roadmap 14.3 — capacité épuisée : on met en attente, on ne surréserve jamais.

    Objectif d'apprentissage
    ------------------------
    Un moteur d'inférence a deux capacités finies : le nombre de séquences qu'il accepte de
    faire tourner ensemble (`max_num_seqs`, qui borne la taille du batch) et le nombre de
    blocs de KV cache disponibles sur la carte (`num_gpu_blocks`, calculé à partir de la
    mémoire restante après chargement des poids, cf. sections 5.1 et 5.2). Admettre une
    requête, c'est réserver ses blocs ; si les blocs manquent, l'admettre quand même
    provoquerait un OOM en pleine génération.

    La règle est donc : on ADMET ou on met EN ATTENTE, jamais on ne surréserve, et rien n'est
    perdu. La file est servie ensuite, en FIFO strict — on ne double pas une requête bloquée,
    même par une petite qui rentrerait, sous peine de famine. C'est la contrepartie du
    continuous batching : il faut un compteur de blocs exact, sinon le moteur promet une
    mémoire qu'il n'a pas.

    Schéma mental
    -------------
        block_size = 4 tokens, pool = 8 blocs, max_num_seqs = 2

            R1 : prompt 10 tokens -> ceil(10/4) = 3 blocs
            R2 : prompt  6 tokens -> ceil(6/4)  = 2 blocs
            R3 : prompt  4 tokens -> ceil(4/4)  = 1 bloc
            R4 : prompt  9 tokens -> ceil(9/4)  = 3 blocs

        1er schedule : running [R1, R2]  (5 blocs pris, 3 libres)  waiting [R3, R4]
        R1 termine   : ses 3 blocs reviennent au pool -> 6 libres
        2e schedule  : running [R2, R3]  (3 blocs pris, 5 libres)  waiting [R4]
        R2 termine   : 3e schedule -> running [R3, R4]  (4 blocs pris, 4 libres)  waiting []

    Ce que ce test vérifie
    ----------------------
    1. la capacité en séquences sature l'admission : 2 requêtes actives, 2 en attente, et
       5 blocs réservés sur 8 donc 3 libres ;
    2. la file est servie dès qu'une place se libère, dans l'ordre d'arrivée : [R2, R3] puis
       [R3, R4], avec 5 puis 4 blocs libres ;
    3. rien n'est jamais perdu ni surréservé : les 4 requêtes sont toujours comptées, et le
       nombre de blocs libres reste entre 0 et 8 ;
    4. la mémoire peut bloquer AVANT la limite de séquences : avec un pool de 4 blocs et 4
       séquences autorisées, une seule requête est admise et 1 bloc libre reste inutilisable.

    API à faire émerger (cible proposée : `src/inference_lab/engine/scheduler.py`)
    ----------------------------------------------------------------------------
        class AdmissionScheduler:
            def __init__(
                self, max_num_seqs: int, num_gpu_blocks: int, block_size: int
            ) -> None: ...
            def add_request(self, request_id: str, prompt_len: int) -> None: ...
            def blocks_needed(self, request_id: str) -> int: ...
            def schedule(self) -> list[str]: ...   # ids admis ET actifs après admission
            def finish(self, request_id: str) -> None: ...
            @property
            def num_free_blocks(self) -> int: ...
            @property
            def running_ids(self) -> list[str]: ...
            @property
            def waiting_ids(self) -> list[str]: ...
            @property
            def finished_ids(self) -> list[str]: ...

        La roadmap indique seulement « scheduler » : on propose le même module que 14.2, avec
        une seconde classe. Au REFACTOR, les deux fusionneront naturellement.

    Indice : `blocks_needed` est le `ceil(prompt_len / block_size)` de la section 14.4, à
    réutiliser plutôt qu'à réécrire. Piège du FIFO strict : la boucle d'admission doit
    s'ARRÊTER au premier candidat qui ne rentre pas (`break`), pas continuer à parcourir la
    file (`continue`) — sinon l'assert 4 tombe, R3 étant admise avant R2. Autre piège :
    rendre les blocs dans `finish()`, sinon le pool fuit et plus rien n'est jamais admis.
    """

    pytest.skip("Roadmap TDD 14.3 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.engine.scheduler import AdmissionScheduler

    # Arrange — un `scheduler` de type `AdmissionScheduler` avec 2 séquences simultanées au
    #           maximum, un pool de 8 blocs de KV cache et des blocs de 4 tokens. Quatre
    #           requêtes ajoutées dans l'ordre "R1", "R2", "R3", "R4", de prompts longs de
    #           respectivement 10, 6, 4 et 9 tokens. Prévoir aussi `saturated`, un second
    #           scheduler autorisant 4 séquences mais ne disposant que de 4 blocs, alimenté
    #           par les trois premières requêtes (prompts de 10, 6 et 4 tokens).

    # Act — appeler `schedule()` une première fois et relever `first_batch`,
    #       `waiting_after_first` et `free_after_first` ; puis terminer "R1", rappeler
    #       `schedule()` et relever `second_batch`, `waiting_after_second` et
    #       `free_after_second` ; puis terminer "R2", rappeler `schedule()` et relever
    #       `third_batch` et `free_after_third`. Appeler enfin `schedule()` une fois sur
    #       `saturated`.

    # Assert 1 — la capacité en séquences sature l'admission, les blocs sont comptés
    assert [scheduler.blocks_needed(rid) for rid in ("R1", "R2", "R3", "R4")] == [3, 2, 1, 3]
    assert first_batch == ["R1", "R2"]
    assert waiting_after_first == ["R3", "R4"]
    assert free_after_first == 3

    # Assert 2 — la file est servie dès qu'une place se libère, dans l'ordre d'arrivée
    assert second_batch == ["R2", "R3"]
    assert waiting_after_second == ["R4"]
    assert free_after_second == 5
    assert third_batch == ["R3", "R4"]
    assert free_after_third == 4

    # Assert 3 — rien n'est perdu, rien n'est surréservé
    assert (
        len(scheduler.running_ids) + len(scheduler.waiting_ids) + len(scheduler.finished_ids) == 4
    )
    assert scheduler.finished_ids == ["R1", "R2"]
    assert 0 <= scheduler.num_free_blocks <= 8

    # Assert 4 — la mémoire bloque avant la limite de séquences (FIFO strict, pas de famine)
    assert saturated.running_ids == ["R1"]
    assert saturated.waiting_ids == ["R2", "R3"]
    assert saturated.num_free_blocks == 1
    assert saturated.blocks_needed("R3") == 1
