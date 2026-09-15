import pytest


@pytest.mark.tdd
def test_gqa_uses_fewer_kv_heads_than_query_heads():
    """
    Objectif
    --------
    Comprendre pourquoi GQA réduit la mémoire KV cache.

    Concepts à comprendre
    ---------------------
    - GQA
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/attention/gqa.py

    Comportement à vérifier
    -----------------------
    num_key_value_heads doit être inférieur à num_attention_heads.

    Assertion attendue
    ------------------
    assert num_kv_heads < num_query_heads

    Hints d'implémentation
    ----------------------
    Utiliser une config miniature inspirée Qwen.

    Critère de réussite
    -------------------
    Le test doit d'abord échouer en RED pour une raison pertinente, puis passer en GREEN
    après l'implémentation minimale dans src/.

    TDD
    ---
    1. supprimer pytest.skip()
    2. construire un Arrange / Act / Assert minimal
    3. écrire l'assertion attendue
    4. obtenir RED
    5. implémenter le minimum dans src/
    6. obtenir GREEN
    7. refactorer sans changer le comportement
    """
    pytest.skip("Roadmap TDD — section pas encore activée")


@pytest.mark.tdd
def test_multiple_query_heads_share_key_value_heads():
    """
    Objectif
    --------
    Vérifier le partage des K/V entre groupes de query heads.

    Concepts à comprendre
    ---------------------
    - GQA
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/attention/gqa.py

    Comportement à vérifier
    -----------------------
    Plusieurs query heads doivent mapper vers le même KV head.

    Assertion attendue
    ------------------
    assert query_head_to_kv_head[0] == query_head_to_kv_head[1]

    Hints d'implémentation
    ----------------------
    Utiliser repeat_interleave ou une fonction de mapping simple.

    Critère de réussite
    -------------------
    Le test doit d'abord échouer en RED pour une raison pertinente, puis passer en GREEN
    après l'implémentation minimale dans src/.

    TDD
    ---
    1. supprimer pytest.skip()
    2. construire un Arrange / Act / Assert minimal
    3. écrire l'assertion attendue
    4. obtenir RED
    5. implémenter le minimum dans src/
    6. obtenir GREEN
    7. refactorer sans changer le comportement
    """
    pytest.skip("Roadmap TDD — section pas encore activée")


@pytest.mark.tdd
def test_mqa_uses_single_key_value_head():
    """
    Objectif
    --------
    Comprendre le cas extrême MQA.

    Concepts à comprendre
    ---------------------
    - MQA
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/attention/gqa.py

    Comportement à vérifier
    -----------------------
    Tous les query heads doivent partager un seul KV head.

    Assertion attendue
    ------------------
    assert num_key_value_heads == 1

    Hints d'implémentation
    ----------------------
    Utiliser repeat_interleave sur la dimension heads.

    Critère de réussite
    -------------------
    Le test doit d'abord échouer en RED pour une raison pertinente, puis passer en GREEN
    après l'implémentation minimale dans src/.

    TDD
    ---
    1. supprimer pytest.skip()
    2. construire un Arrange / Act / Assert minimal
    3. écrire l'assertion attendue
    4. obtenir RED
    5. implémenter le minimum dans src/
    6. obtenir GREEN
    7. refactorer sans changer le comportement
    """
    pytest.skip("Roadmap TDD — section pas encore activée")


