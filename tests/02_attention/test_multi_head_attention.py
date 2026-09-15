import pytest


@pytest.mark.tdd
def test_hidden_dimension_is_split_across_attention_heads():
    """
    Objectif
    --------
    Comprendre comment hidden_size est réparti entre les heads.

    Concepts à comprendre
    ---------------------
    - MHA
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/attention/mha.py

    Comportement à vérifier
    -----------------------
    hidden_size doit être num_heads * head_dim.

    Assertion attendue
    ------------------
    assert hidden_size == num_heads * head_dim

    Hints d'implémentation
    ----------------------
    Utiliser reshape [B,T,H,D] puis transpose [B,H,T,D].

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
def test_attention_heads_are_concatenated_back_to_hidden_dimension():
    """
    Objectif
    --------
    Vérifier que les têtes reviennent vers hidden_size après attention.

    Concepts à comprendre
    ---------------------
    - MHA
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/attention/mha.py

    Comportement à vérifier
    -----------------------
    La concaténation des heads doit redonner [B,T,hidden_size].

    Assertion attendue
    ------------------
    assert out.shape == (B, T, hidden_size)

    Hints d'implémentation
    ----------------------
    Utiliser transpose puis contiguous().view(...).

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


