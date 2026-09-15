import pytest


@pytest.mark.tdd
def test_attention_probabilities_sum_to_one():
    """
    Objectif
    --------
    Vérifier que softmax produit une distribution sur les keys.

    Concepts à comprendre
    ---------------------
    - softmax
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/attention/naive.py

    Comportement à vérifier
    -----------------------
    La somme des probabilités sur key_length doit valoir 1.

    Assertion attendue
    ------------------
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones_like(...))

    Hints d'implémentation
    ----------------------
    Utiliser torch.softmax(dim=-1).

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
def test_masked_positions_receive_zero_probability():
    """
    Objectif
    --------
    S’assurer que les positions masquées ne contribuent pas à la sortie.

    Concepts à comprendre
    ---------------------
    - softmax
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/attention/naive.py

    Comportement à vérifier
    -----------------------
    Après softmax, les probabilités des positions masquées doivent être nulles.

    Assertion attendue
    ------------------
    assert probs[masked_positions].eq(0).all()

    Hints d'implémentation
    ----------------------
    Appliquer mask avant softmax avec -inf.

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


