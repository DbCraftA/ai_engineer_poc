"""Section 14.8 — chunked prefill : découper un long prompt pour ne pas geler le decode.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/14_engine/test_chunked_prefill.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(1000 tokens en chunks de 256 -> [256, 256, 256, 232], budget de 256 tokens par étape) :
c'est volontaire. Un test doit énoncer la vérité attendue, pas la recalculer avec la même
formule que le code testé.

Dernière pièce de l'ordonnanceur. Le prefill et le decode ne se ressemblent pas : le prefill
traite des milliers de tokens d'un coup (calcul intensif), le decode un seul token par
séquence (mémoire intensive, cf. section 13). Un long prefill monopolise donc l'itération et
gèle le decode de toutes les requêtes en cours, ce qui se voit immédiatement sur le TPOT.
Découper le prefill en chunks bornés remet les deux régimes dans un budget de tokens commun.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices. Même raison pour l'import
# du module cible : c'est ton code d'`Act` qui l'appellera, le linter le voit donc inutilisé.
# ruff: noqa: F401, F821

import pytest


@pytest.mark.tdd
def test_long_prefill_can_be_split_into_multiple_chunks():
    """Roadmap 14.8 — un prompt long devient une suite de chunks bornés, sans perte de token.

    Objectif d'apprentissage
    ------------------------
    Un prefill de 1000 tokens et une étape de decode ne coûtent pas du tout la même chose : le
    premier est un gros GEMM (des milliers de tokens en parallèle), la seconde relit surtout
    le KV cache pour un token par séquence. Si l'ordonnanceur laisse le prefill occuper une
    itération entière, les séquences déjà en cours de génération n'avancent pas pendant tout
    ce temps : leur TPOT (section 13.2) prend un pic très visible, alors même que le débit
    global paraît bon.

    Le chunked prefill découpe donc le prompt en tranches d'au plus `chunk_size` tokens, et
    l'ordonnanceur mélange dans une même itération une tranche de prefill et les tokens de
    decode des séquences actives, sous un budget total `max_num_batched_tokens`. Le decode
    n'attend plus jamais plus d'une tranche. La correction du découpage est critique : la
    somme des chunks doit valoir EXACTEMENT la longueur du prompt, sinon des tokens sont
    oubliés ou traités deux fois, et le KV cache est faux.

    Schéma mental
    -------------
        prompt de 1000 tokens, chunk_size = 256

            chunk 0 : 256      chunk 1 : 256      chunk 2 : 256      chunk 3 : 232
            4 chunks = ceil(1000/256), somme = 768 + 232 = 1000, aucun chunk > 256

        budget max_num_batched_tokens = 256, 2 séquences en cours de decode :

            avec chunked prefill : 254 tokens de prefill + 2 de decode = 256  -> decode servi
            sans chunked prefill : 1000 tokens de prefill + 2 de decode = 1002, presque 4 fois
                                   le budget, et le decode des 2 séquences attend cette étape

    Ce que ce test vérifie
    ----------------------
    1. le découpage attendu, chunk par chunk : [256, 256, 256, 232], soit 4 chunks ;
    2. la conservation des tokens : la somme des chunks vaut exactement 1000, les trois
       premiers sont pleins et le dernier vaut le reste, 232, strictement plus petit que la
       taille de chunk ;
    3. les cas limites du découpage : multiple exact, prompt plus court qu'un chunk, prompt
       vide, et jamais aucun chunk au-dessus de la borne ;
    4. l'intérêt réel : sous un budget de 256 tokens et 2 séquences en decode, l'étape
       accueille 254 tokens de prefill ET les 2 tokens de decode, donc le decode n'est pas
       bloqué par le prompt long.

    API à faire émerger (cible roadmap : `src/inference_lab/engine/scheduler.py`)
    ---------------------------------------------------------------------------
        def split_prefill_into_chunks(prompt_len: int, chunk_size: int) -> list[int]: ...
        def plan_chunked_step(
            prompt_len: int, num_decode_seqs: int, max_num_batched_tokens: int
        ) -> tuple[int, int]: ...            # (tokens de prefill, tokens de decode)

        La roadmap indique « scheduler » : on propose le module concret déjà utilisé par les
        tests 14.2 et 14.3, `src/inference_lab/engine/scheduler.py`.

    Indice : le decode est prioritaire dans `plan_chunked_step` — on lui réserve d'abord un
    token par séquence active, et le prefill prend ce qui reste du budget. Pièges : renvoyer
    une liste de longueurs égales en divisant 1000 par 4 (250, 250, 250, 250) n'est PAS le
    comportement attendu, les chunks se remplissent au maximum puis le dernier prend le
    reste ; et un prompt vide doit rendre une liste vide, pas `[0]`, sinon l'ordonnanceur
    planifie une étape qui ne calcule rien.
    """

    pytest.skip("Roadmap TDD 14.8 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.engine.scheduler import plan_chunked_step, split_prefill_into_chunks

    # Arrange — quatre entiers, aucun tenseur : `prompt_len` = 1000 tokens (le prompt long à
    #           découper), `chunk_size` = 256 tokens par tranche, `max_num_batched_tokens` =
    #           256 (le budget de tokens d'une itération du moteur) et `num_decode_seqs` = 2,
    #           le nombre de séquences déjà en cours de génération qu'il ne faut pas geler.

    # Act — découper le prefill du prompt en `chunks`, puis demander à l'ordonnanceur la
    #       composition d'une étape mêlant prefill et decode dans `plan`.

    # Assert 1 — le découpage attendu, chunk par chunk
    assert chunks == [256, 256, 256, 232]
    assert len(chunks) == 4

    # Assert 2 — aucun token perdu ni compté deux fois
    assert sum(chunks) == 1000
    assert chunks[:3] == [256, 256, 256]
    assert chunks[-1] == 232
    assert chunks[-1] < chunk_size

    # Assert 3 — les cas limites, et la borne jamais dépassée
    assert max(chunks) == chunk_size
    assert split_prefill_into_chunks(512, chunk_size) == [256, 256]
    assert split_prefill_into_chunks(200, chunk_size) == [200]
    assert split_prefill_into_chunks(0, chunk_size) == []

    # Assert 4 — le decode partage l'étape avec le prefill au lieu de l'attendre
    assert plan == (254, 2)
    assert sum(plan) == 256
    assert plan[1] == num_decode_seqs
    assert prompt_len + num_decode_seqs == 1002
