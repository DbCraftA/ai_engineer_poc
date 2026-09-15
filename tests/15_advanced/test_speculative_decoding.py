"""Section 15.4 / 15.5 — decoding spéculatif : accélérer sans changer une seule sortie.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/15_advanced/test_speculative_decoding.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(3 propositions par étape, 8 tokens générés en 2 appels au modèle cible, 4 acceptations sur
6 propositions, taux 2/3) : c'est volontaire. Un test doit énoncer la vérité attendue, pas la
recalculer avec la même formule que le code testé.

Le decoding spéculatif est la dernière optimisation de la roadmap et la plus contre-intuitive :
un petit modèle *draft* propose plusieurs tokens d'avance, le modèle cible les VÉRIFIE tous en
une seule passe, garde le plus long préfixe correct et jette le reste. Le gain est de faire
sortir plusieurs tokens d'un seul appel au gros modèle, dont le decode est limité par la
lecture des poids (section 7) et non par le calcul. La contrainte est absolue et fait l'objet
de 15.4 : la sortie doit être celle du modèle cible SEUL, sinon l'optimisation est un bug —
même exigence qu'en 4.6 pour le KV cache. Le test 15.5 mesure ce qui varie vraiment selon la
qualité du draft : la vitesse, jamais le texte.

Roadmap et modules cibles : `tests/ROADMAP.md` (colonne « Code src cible »).
"""

# Les blocs `Assert` référencent volontairement des variables qui n'existent pas encore :
# c'est à toi de les créer dans la partie `Arrange` de chaque test. On désactive donc le
# contrôle « nom non défini » du linter sur ce fichier d'exercices. Même raison pour l'import
# du module cible : c'est ton code d'`Act` qui l'appellera, le linter le voit donc inutilisé.
# ruff: noqa: F401, F821

import pytest
import torch


@pytest.mark.tdd
def test_verified_speculative_tokens_match_target_model_distribution():
    """Roadmap 15.4 — la vérification garantit les tokens du modèle cible, à l'id près.

    Objectif d'apprentissage
    ------------------------
    C'est LA propriété fondamentale du decoding spéculatif : le draft ne décide de rien, il
    ne fait que des propositions. Le modèle cible relit chaque proposition et n'accepte qu'un
    PRÉFIXE : dès qu'une proposition diffère de ce qu'il aurait produit, elle est rejetée,
    ainsi que toutes celles qui suivent — et le modèle cible fournit lui-même le token de
    remplacement (le token « bonus »). Chaque étape avance donc d'au moins un token, quelle
    que soit la qualité du draft.

    En greedy avec deux modèles déterministes, l'énoncé est exact : la suite d'ids produite
    doit être IDENTIQUE à celle du modèle cible seul, comparable avec `torch.equal` sans
    tolérance (même raisonnement qu'en 4.6 : l'argmax absorbe le bruit numérique). En
    échantillonnage, la même garantie existe sur la DISTRIBUTION, via l'acceptation-rejet
    modifiée — c'est le cœur de l'article de Leviathan et al., et la raison pour laquelle
    cette technique est considérée comme sans perte, contrairement à la quantification
    de 15.1 / 15.2.

    Corollaire à retenir : un draft médiocre coûte du calcul gaspillé, jamais une sortie
    fausse. Un test qui échoue ici ne dit donc pas « le draft est mauvais », il dit « la
    vérification, la position RoPE (4.7) ou la troncature du cache est buguée ».

    Schéma mental
    -------------
        prompt de 3 tokens, 8 tokens à générer, 3 propositions par étape (k = 3)

        baseline greedy : 8 passes du modèle cible, 1 token chacune

        draft ALIGNÉ (copie du modèle cible) — tout est accepté :
            étape 1 : draft d1 d2 d3 -> cible vérifie -> 3 acceptés + 1 bonus = 4 tokens
            étape 2 : draft d4 d5 d6 -> cible vérifie -> 3 acceptés + 1 bonus = 4 tokens
            2 appels au modèle cible pour 8 tokens          ids identiques à la baseline

        draft TOUJOURS FAUX (scripté pour se tromper dès sa 1re proposition) :
            chaque étape : 0 accepté + 1 bonus = 1 token
            8 appels au modèle cible pour 8 tokens          ids identiques à la baseline

    Ce que ce test vérifie
    ----------------------
    1. l'égalité stricte des ids avec la génération greedy du modèle cible seul, en shape
       (1, 11) — 3 tokens de prompt et 8 générés — et en dtype entier ;
    2. la comptabilité de l'étape avec un draft aligné : 3 acceptations par étape, 6 tokens
       acceptés sur 6 proposés, prompt intact en préfixe ;
    3. le gain : 2 appels au modèle cible au lieu de 8, pour un résultat identique ;
    4. l'invariant ne dépend PAS du draft : avec un draft toujours faux, les ids sont encore
       exactement ceux de la baseline, mais aucun token n'est accepté et il faut 8 appels —
       pire cas en vitesse, jamais en qualité.

    API à faire émerger (cible roadmap « future speculative decoding », cible proposée :
    `src/inference_lab/inference/speculative.py`, à côté de `generation.py` de 4.6)
    -----------------------------------------------------------------------------------
        @dataclass
        class SpeculativeResult:
            output_ids: torch.Tensor      # (batch, prompt + max_new_tokens)
            accepted_per_step: list[int]
            proposed_tokens: int
            accepted_tokens: int
            target_calls: int             # passes du modèle cible pendant le decode

        def speculative_generate(
            target_model: torch.nn.Module,
            draft_model: torch.nn.Module,
            prompt_ids: torch.Tensor,
            max_new_tokens: int,
            num_speculative_tokens: int = 3,
        ) -> SpeculativeResult: ...

    Indice : la vérification est un `argmax` sur les logits du modèle cible aux positions des
    propositions, puis la longueur du préfixe commun avec les propositions du draft
    (`(proposed != verified).int().argmax()` en gérant le cas « aucun désaccord »). Pièges :
    ne jamais accepter un token après un rejet (le préfixe s'arrête au PREMIER désaccord),
    tronquer le KV cache du modèle cible et celui du draft à la longueur acceptée, et ne pas
    dépasser `max_new_tokens` avec le token bonus de la dernière étape.
    """

    pytest.skip("Roadmap TDD 15.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.inference.generation import generate
    from inference_lab.inference.speculative import speculative_generate

    # Arrange — le modèle jouet causal déterministe des sections 4.x : batch=1, num_layers=2,
    #           num_heads=2, num_kv_heads=1, head_dim=4, vocab=16, `eval()`, sans dropout,
    #           float32, seed fixée, logits sans ex aequo en tête. `target_model` est ce
    #           modèle, `prompt_ids` un tenseur (1, 3) de `torch.long` aux ids distincts.
    #           Deux drafts : `aligned_draft`, une copie exacte de `target_model` (donc
    #           toujours d'accord avec lui), et `always_wrong_draft`, un draft scripté qui
    #           propose systématiquement, dès sa première proposition, un id différent de
    #           l'argmax du modèle cible.

    # Act — générer 8 tokens en greedy avec `generate` sur `target_model` seul
    #       (`target_only_ids`, `baseline_target_calls` = une passe par token), puis appeler
    #       `speculative_generate` avec 3 propositions par étape, une fois avec
    #       `aligned_draft` (`aligned`) et une fois avec `always_wrong_draft` (`misaligned`).

    # Assert 1 — l'invariant : exactement les ids du modèle cible seul
    assert torch.equal(aligned.output_ids, target_only_ids)
    assert aligned.output_ids.shape == (1, 11)
    assert aligned.output_ids.dtype is torch.long

    # Assert 2 — la comptabilité d'un draft parfaitement aligné
    assert aligned.accepted_per_step == [3, 3]
    assert aligned.proposed_tokens == 6
    assert aligned.accepted_tokens == 6
    assert torch.equal(aligned.output_ids[:, :3], prompt_ids)

    # Assert 3 — le gain : 4 fois moins d'appels au gros modèle pour le même texte
    assert baseline_target_calls == 8
    assert aligned.target_calls == 2

    # Assert 4 — un draft toujours faux ralentit, il ne change rien à la sortie
    assert torch.equal(misaligned.output_ids, target_only_ids)
    assert misaligned.accepted_tokens == 0
    assert misaligned.proposed_tokens == 24
    assert misaligned.target_calls == 8


@pytest.mark.tdd
def test_acceptance_rate_is_computed_from_verified_draft_tokens():
    """Roadmap 15.5 — le taux d'acceptation mesure la vitesse, pas la qualité.

    Objectif d'apprentissage
    ------------------------
    Puisque la sortie est figée par 15.4, la seule métrique qui reste à surveiller est le taux
    d'acceptation :

        taux = tokens acceptés / tokens proposés            0 <= taux <= 1

    C'est lui qui décide si le decoding spéculatif rapporte quelque chose. Chaque étape coûte
    une passe du modèle cible (plus k passes du petit draft, supposées bon marché) et rend
    `acceptés + 1` tokens. Le taux se traduit donc directement en tokens par appel au modèle
    cible, c'est-à-dire en TPOT (section 13.2) :

        espérance théorique, acceptations i.i.d. de probabilité a :
            tokens_par_appel = 1 + a + a^2 + ... + a^k = (1 - a^(k+1)) / (1 - a)

    Avec k = 3 : 1 token par appel si a = 0 (aucun gain, on a juste payé le draft), 4 si
    a = 1. Un taux faible n'est jamais une erreur de correction — c'est un draft mal choisi,
    un domaine trop différent, ou un k trop grand pour la qualité du draft. Le réglage de k
    est un arbitrage : plus de propositions par étape, mais une probabilité décroissante que
    la dernière survive.

    Attention à ne pas confondre les deux chiffres du test : la mesure d'un scénario donné
    (6 tokens en 2 appels, soit 3,0 tokens par appel) et l'espérance théorique du même taux
    (2,407 tokens par appel), qui moyenne sur toutes les répartitions possibles des rejets.

    Schéma mental
    -------------
        k = 3 propositions par étape, 2 étapes, draft scripté

            étape 1 : 3 propositions -> 3 acceptées + 1 bonus = 4 tokens
            étape 2 : 3 propositions -> 1 acceptée  + 1 bonus = 2 tokens
                                        (la 2e proposition est fausse, la 3e est jetée)

            proposés = 6      acceptés = 4      taux = 4 / 6 = 0,6667
            tokens générés = 4 + 2 = 6 en 2 appels cible -> 3,0 tokens par appel

        espérance pour a = 2/3 et k = 3 : 1 + 2/3 + 4/9 + 8/27 = 65/27 = 2,407
        bornes : a = 0 -> 1,0 token par appel      a = 1 -> 4,0 tokens par appel

    Ce que ce test vérifie
    ----------------------
    1. le décompte du scénario : 6 propositions, 4 acceptations réparties en [3, 1], et un
       taux de 4/6 ≈ 0,6667 ;
    2. les bornes et les cas dégénérés : le taux reste dans [0, 1], vaut 1.0 si tout est
       accepté, 0.0 si tout est rejeté, et 0.0 par convention si rien n'a été proposé ;
    3. le débit mesuré du scénario : 6 tokens générés pour 2 appels au modèle cible, soit
       3,0 tokens par appel, contre 1 token par appel sans spéculation ;
    4. l'espérance théorique associée à ce taux (65/27 ≈ 2,407 pour k = 3), encadrée par
       1,0 en a = 0 et 4,0 en a = 1 : le taux est monotone en vitesse, et jamais en qualité.

    API à faire émerger (cible roadmap « future speculative decoding », cible proposée :
    `src/inference_lab/inference/speculative.py`)
    -----------------------------------------------------------------------------------
        def acceptance_rate(accepted_tokens: int, proposed_tokens: int) -> float: ...

        def expected_tokens_per_target_call(
            num_speculative_tokens: int,
            acceptance_rate: float,
        ) -> float: ...

        et le `SpeculativeResult` de 15.4, dont `accepted_per_step`, `proposed_tokens`,
        `accepted_tokens` et `target_calls` fournissent les compteurs mesurés.

    Indice : `expected_tokens_per_target_call` est une somme géométrique — écris-la comme une
    somme de puissances plutôt qu'avec la forme fermée, qui divise par zéro en a = 1. Pièges :
    `proposed_tokens` compte TOUTES les propositions du draft, y compris celles jetées après
    un rejet (sinon le taux vaut toujours 1) ; et `acceptance_rate(0, 0)` doit rendre 0.0 au
    lieu de lever `ZeroDivisionError`, car c'est l'état initial de n'importe quel compteur.
    """

    pytest.skip("Roadmap TDD 15.5 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.inference.speculative import (
        acceptance_rate,
        expected_tokens_per_target_call,
        speculative_generate,
    )

    # Arrange — le même modèle jouet déterministe qu'en 15.4 (`target_model`, `prompt_ids` de
    #           shape (1, 3)), et `scripted_draft`, un draft scripté qui reproduit exactement
    #           le scénario du schéma : à la première étape ses 3 propositions sont celles du
    #           modèle cible, à la deuxième seule la première proposition est correcte, les
    #           deux suivantes sont fausses. Le nombre de propositions par étape est 3, et on
    #           s'arrête après 6 tokens générés.

    # Act — appeler `speculative_generate` avec ce draft pour obtenir `stats`, puis en déduire
    #       `measured_rate` (le taux d'acceptation des compteurs de `stats`) et
    #       `measured_tokens_per_call` (tokens générés / appels au modèle cible).

    # Assert 1 — le décompte du scénario et le taux attendu
    assert stats.accepted_per_step == [3, 1]
    assert stats.proposed_tokens == 6
    assert stats.accepted_tokens == 4
    assert measured_rate == pytest.approx(0.6667, rel=1e-3)
    assert measured_rate == pytest.approx(2 / 3, rel=1e-9)

    # Assert 2 — bornes et cas dégénérés d'un taux
    assert 0.0 <= measured_rate <= 1.0
    assert acceptance_rate(6, 6) == pytest.approx(1.0, rel=1e-9)
    assert acceptance_rate(0, 6) == pytest.approx(0.0, abs=1e-12)
    assert acceptance_rate(0, 0) == pytest.approx(0.0, abs=1e-12)

    # Assert 3 — le débit mesuré : 6 tokens pour 2 appels au modèle cible
    assert stats.output_ids.shape == (1, 9)
    assert stats.target_calls == 2
    assert measured_tokens_per_call == pytest.approx(3.0, rel=1e-9)

    # Assert 4 — l'espérance théorique du même taux, et ses deux bornes en k = 3
    assert expected_tokens_per_target_call(3, 2 / 3) == pytest.approx(65 / 27, rel=1e-9)
    assert expected_tokens_per_target_call(3, 0.0) == pytest.approx(1.0, rel=1e-9)
    assert expected_tokens_per_target_call(3, 1.0) == pytest.approx(4.0, rel=1e-9)
