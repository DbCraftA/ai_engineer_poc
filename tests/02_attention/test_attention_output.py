import pytest


@pytest.mark.tdd
def test_attention_output_is_weighted_sum_of_values():
    """
    Objectif
    --------
    Relier les probabilités d’attention à la somme pondérée de V.

    Concepts à comprendre
    ---------------------
    - weighted V
    - shapes et layout lorsque pertinent
    - différence entre tenseur temporaire, paramètre, buffer et sortie
    - rôle dans l'inférence LLM lorsque pertinent

    Code cible
    ----------
    src/inference_lab/nn/attention/naive.py

    Comportement à vérifier
    -----------------------
    La sortie doit être weights @ v.

    Assertion attendue
    ------------------
    torch.testing.assert_close(output, weights @ v)

    Hints d'implémentation
    ----------------------
    Utiliser torch.matmul sur les deux dernières dimensions.

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


