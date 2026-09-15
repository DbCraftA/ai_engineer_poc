"""Section 3.10 — boucle de génération : un token par itération, arrêt sur EOS.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/03_models/test_generation.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(4 + 3 = 7 tokens, (1, 5), ...) : c'est volontaire. Un test doit énoncer la vérité attendue,
pas la recalculer avec la même formule que le code testé.

Modèle jouet de la section 3, constant dans tout le fichier : `vocab_size=16`, `hidden=8`,
`num_layers=2`, `num_heads=2`, `seq=4` (longueur du prompt), `batch=1`. Cette boucle est la
version NAÏVE : à chaque itération on repasse toute la séquence dans le modèle. C'est
volontairement inefficace — la section 4 mesurera ce coût quadratique puis le supprimera avec
le KV cache, en exigeant une sortie identique token par token.

Convention retenue pour l'arrêt : le token EOS généré est CONSERVÉ dans la sortie, puis la
boucle s'arrête immédiatement (c'est le comportement de `transformers`).

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices.
# ruff: noqa: F821

import pytest
import torch


@pytest.mark.tdd
@pytest.mark.model
def test_generation_appends_one_token_per_iteration():
    """Roadmap 3.10 — générer, c'est boucler : un token produit et réinjecté par itération.

    Objectif d'apprentissage
    ------------------------
    Un LLM ne produit pas une phrase, il produit UN token. La phrase vient de la boucle
    autorégressive : on prend les logits de la dernière position, on choisit un id, on
    l'ajoute à la séquence, et on recommence avec la séquence allongée. La longueur finale
    est donc parfaitement prévisible : `len(prompt) + max_new_tokens` (sauf arrêt sur EOS).
    C'est aussi ici que naît le vocabulaire prefill/decode : la première itération traite
    les 4 tokens du prompt, les suivantes n'en ajoutent qu'un — et refont pourtant tout le
    travail dans cette version naïve.

    Schéma mental
    -------------
        prompt (batch=1, seq=4)                        max_new_tokens = 3

        itération 1 : (1, 4) -> logits (1, 4, 16) -> logits[:, -1, :] -> +1 id -> (1, 5)
        itération 2 : (1, 5) -> logits (1, 5, 16) -> logits[:, -1, :] -> +1 id -> (1, 6)
        itération 3 : (1, 6) -> logits (1, 6, 16) -> logits[:, -1, :] -> +1 id -> (1, 7)

        sortie (1, 7) = 4 + 3, et les 4 premières colonnes SONT le prompt d'origine

    Ce que ce test vérifie
    ----------------------
    1. la longueur finale vaut exactement `len(prompt) + max_new_tokens = 4 + 3 = 7`, batch
       inchangé, et la sortie est un tenseur d'ids `torch.long` ;
    2. le prompt est conservé tel quel en préfixe : la génération ajoute, elle ne réécrit pas ;
    3. tous les ids générés sont des indices valides du vocabulaire, dans `[0, 16)` ;
    4. la boucle avance d'exactement un token par itération (`max_new_tokens=1` rend `(1, 5)`)
       et reste déterministe en greedy : deux appels donnent la même séquence.

    API à faire émerger (cible roadmap : `src/inference_lab/inference/generation.py`)
    -------------------------------------------------------------------------------
        def generate(
            model: torch.nn.Module,
            token_ids: torch.Tensor,
            max_new_tokens: int,
            eos_token_id: int | None = None,
        ) -> torch.Tensor: ...

    Indice : `for _ in range(max_new_tokens)` ; à l'intérieur, `logits = model(token_ids)`,
    puis le token greedy de `logits[:, -1, :]` (section 3.6), puis
    `token_ids = torch.cat([token_ids, next_token.unsqueeze(-1)], dim=-1)`. Pièges : prendre
    `logits[:, -1, :]` et non `logits[:, 0, :]`, garder `dtype=torch.long` lors du `cat`, ne
    pas modifier le tenseur d'entrée en place, et envelopper la boucle dans `torch.no_grad()`.
    """

    pytest.skip("Roadmap TDD 3.10 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.inference.generation import generate

    # Arrange — fixer la graine (`torch.manual_seed(0)`) puis construire :
    #           - `model`, le `MinimalTransformer` de la section 3.4 (vocab_size=16,
    #             hidden_size=8, num_layers=2, num_heads=2), en mode évaluation ;
    #           - `prompt`, un tenseur `torch.long` de shape (batch=1, seq=4) dont les ids sont
    #             dans [0, 16) ;
    #           - `max_new_tokens = 3`, et aucun id d'arrêt (`eos_token_id` laissé à None) pour
    #             que la boucle aille jusqu'au bout.

    # Act — appeler la génération greedy sur `model` et `prompt` pour obtenir `output`.

    # Assert 1 — longueur finale = longueur du prompt + tokens demandés
    assert output.shape == (1, 7)
    assert output.dtype is torch.long

    # Assert 2 — le prompt est conservé en préfixe
    assert torch.equal(output[:, :4], prompt)

    # Assert 3 — les tokens générés sont des ids valides du vocabulaire
    assert int(output.min()) >= 0
    assert int(output.max()) < 16

    # Assert 4 — un seul token par itération, et un greedy reproductible
    assert generate(model, prompt, max_new_tokens=1).shape == (1, 5)
    assert torch.equal(generate(model, prompt, max_new_tokens=3), output)


@pytest.mark.tdd
@pytest.mark.model
def test_generation_stops_on_eos():
    """Roadmap 3.10 — l'id EOS interrompt la boucle avant d'avoir consommé le budget de tokens.

    Objectif d'apprentissage
    ------------------------
    `max_new_tokens` est un plafond, pas un objectif. Un modèle signale la fin de sa réponse
    en produisant un token spécial (`<|endoftext|>`, `<|im_end|>`...) : la boucle doit le
    détecter et sortir. Sans cette condition, un moteur continue de facturer du calcul après
    la fin utile de la réponse, et le serveur ne peut jamais libérer les blocs de KV cache
    d'une requête terminée. C'est la seule raison pour laquelle les longueurs de séquences
    d'un batch divergent, ce qui rendra le batching de la section 6 non trivial.

    Schéma mental
    -------------
        prompt (batch=1, seq=4), SANS aucun eos_token_id dedans
        modèle truqué : quel que soit l'état, argmax(logits[:, -1, :]) == eos_token_id
        max_new_tokens = 3

        itération 1 : ajoute eos -> (1, 5) -> ARRÊT
        itérations 2 et 3 : jamais exécutées

        sortie (1, 5), dernier id == eos_token_id, 5 < 4 + 3 = 7

    Ce que ce test vérifie
    ----------------------
    1. la boucle s'arrête dès l'EOS : un seul token ajouté au lieu de 3, soit une sortie
       `(1, 5)`, strictement plus courte que `len(prompt) + max_new_tokens = 7` — et augmenter
       le budget de tokens ne change rien, puisque l'arrêt vient du modèle, pas du compteur ;
    2. le token EOS est conservé comme dernier id de la sortie (convention du fichier) ;
    3. l'EOS n'apparaît qu'une fois : rien n'est généré après lui ;
    4. le prompt reste intact en préfixe, même sur une génération interrompue.

    API à faire émerger (cible roadmap : `src/inference_lab/inference/generation.py`)
    -------------------------------------------------------------------------------
        def generate(
            model: torch.nn.Module,
            token_ids: torch.Tensor,
            max_new_tokens: int,
            eos_token_id: int | None = None,
        ) -> torch.Tensor: ...

    Indice : concatène le token choisi PUIS teste `if eos_token_id is not None and
    int(next_token) == eos_token_id: break`. Pour le modèle truqué, un petit
    `torch.nn.Module` dont le `forward(token_ids)` renvoie un tenseur de logits
    `(batch, seq, 16)` nul partout sauf une valeur positive dans la colonne `eos_token_id`
    suffit : la boucle de génération n'exige rien de plus qu'un appelable rendant des logits.
    """

    pytest.skip("Roadmap TDD 3.10 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.inference.generation import generate

    # Arrange — construire :
    #           - `eos_token_id`, un id valide du vocabulaire (dans [0, 16)) ;
    #           - `eos_model`, un `torch.nn.Module` minuscule dont le forward accepte des ids
    #             de shape (batch, seq) et renvoie des logits de shape (batch, seq, 16) dont
    #             l'argmax sur la dernière dimension est TOUJOURS `eos_token_id` ;
    #           - `prompt`, un tenseur `torch.long` de shape (batch=1, seq=4) d'ids dans
    #             [0, 16) qui ne contient AUCUNE occurrence de `eos_token_id` ;
    #           - `max_new_tokens = 3`, volontairement plus grand que nécessaire.

    # Act — appeler la génération sur `eos_model` et `prompt` en passant `eos_token_id`,
    #       pour obtenir `output`.

    # Assert 1 — arrêt anticipé : un seul token ajouté sur les 3 autorisés
    assert output.shape == (1, 5)
    assert output.shape[1] < 4 + 3
    assert torch.equal(
        generate(eos_model, prompt, max_new_tokens=10, eos_token_id=eos_token_id), output
    )

    # Assert 2 — le token EOS est conservé, et c'est le dernier
    assert int(output[0, -1]) == eos_token_id

    # Assert 3 — rien n'est généré après l'EOS
    assert int((output[0] == eos_token_id).sum()) == 1

    # Assert 4 — le prompt reste intact
    assert torch.equal(output[:, :4], prompt)
