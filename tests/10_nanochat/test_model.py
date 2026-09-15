import pytest


@pytest.mark.tdd
@pytest.mark.model
def test_nanochat_like_model_outputs_expected_logits_shape():
    """
    Objectif
    --------
    Dans ce test, l'objectif est de vérifier que pour un cas minimal lié à `nanochat like model outputs expected logits shape`, le comportement attendu est observable directement dans le test avant d'être extrait dans le code source.

    Pourquoi c'est important
    ------------------------
    Ce test sert de contrat TDD. Il doit expliquer ce que l'on veut apprendre, quel comportement doit exister, et quelle API minimale devra émerger dans `src/` lorsque la section sera activée.

    Comportement à vérifier
    -----------------------
    Étant donné un exemple volontairement petit qui illustre `nanochat like model outputs expected logits shape`, quand on exécutera l'opération cible, alors le résultat devra correspondre exactement à l'attendu décrit par le nom du test.

    Assertion attendue
    ------------------
    `assert actual == expected` avec `expected` remplacé par la valeur concrète attendue pour `nanochat like model outputs expected logits shape`.

    Hints d'implémentation
    ----------------------
    Utiliser de petits tenseurs CPU, lire `.shape`, puis comparer avec `torch.Size([...])`. Le code cible indiqué par la roadmap est `src/inference_lab/models/nanochat_like/model.py`.

    TDD
    ---
    1. supprimer `pytest.skip(...)` ;
    2. conserver ou affiner l'Arrange / Act / Assert ci-dessous ;
    3. obtenir RED si le code cible n'existe pas encore ou si le comportement est faux ;
    4. implémenter le minimum dans `src/` ;
    5. obtenir GREEN ;
    6. refactorer sans changer le comportement.
    """

    pytest.skip("Roadmap TDD — section pas encore activée")

    # Arrange
    # Construire ici un exemple minimal qui rend visible le comportement :
    # `test_nanochat_like_model_outputs_expected_logits_shape`.
    # Remplacer cette valeur texte par une vraie valeur attendue lors de l'activation.
    expected = "nanochat like model outputs expected logits shape"

    # Act
    # Appeler ici la fonction ou méthode cible qui émergera de `src/inference_lab/models/nanochat_like/model.py`.
    # Remplacer cette valeur texte par le résultat réellement observé.
    actual = "nanochat like model outputs expected logits shape"

    # Assert
    assert actual == expected


