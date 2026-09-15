"""Section 3.6 / 3.7 / 3.8 / 3.9 — échantillonnage : des logits au token choisi.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/03_models/test_sampling.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
(12 exclus sur 16, exactement 4 candidats, probabilité 0.0, ...) : c'est volontaire. Un test
doit énoncer la vérité attendue, pas la recalculer avec la même formule que le code testé.

Modèle jouet de la section 3, constant dans tout le fichier : `vocab_size=16`, `hidden=8`,
`num_layers=2`, `num_heads=2`, `seq=4`, `batch=1`. Ici on ne travaille QUE sur la dernière
ligne de logits, `(batch=1, vocab_size=16)` : c'est exactement ce que produit un pas de decode.
Tous ces filtres sont des transformations de logits vers logits (jamais vers probabilités),
ce qui permet de les composer : temperature, puis top-k, puis top-p, puis tirage.

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
def test_greedy_sampling_selects_highest_logit():
    """Roadmap 3.6 — le décodage greedy est un `argmax` : déterministe et invariant par décalage.

    Objectif d'apprentissage
    ------------------------
    Greedy est la stratégie de référence du projet, précisément parce qu'elle est
    déterministe : c'est la seule façon de comparer notre moteur à Hugging Face token par
    token (`tests/qwen/`) ou de vérifier qu'activer le KV cache ne change pas la sortie
    (section 4.6). Elle ne regarde que la position d'un maximum, donc elle ignore l'échelle
    des logits : ajouter une constante à tous les logits ne change rien, ce qui explique
    aussi pourquoi la temperature n'a aucun effet sur greedy.

    Schéma mental
    -------------
        logits (batch=1, vocab_size=16)
        [[ l0, l1, ..., l15 ]]        un maximum UNIQUE, à un index i
             |
             v  argmax(dim=-1)
        token (1,)  dtype long, valeur i, dans [0, 16)

        argmax(logits) == argmax(logits + 10.0)     -> insensible au décalage

    Ce que ce test vérifie
    ----------------------
    1. la sortie est un tenseur d'ids `torch.long` de shape `(batch,) = (1,)`, dans `[0, 16)` ;
    2. l'id choisi est celui du logit maximum, avec `logits.argmax(dim=-1)` comme oracle ;
    3. deux appels sur les mêmes logits donnent le même token : aucune source d'aléa ;
    4. ajouter une constante à tous les logits ne change pas le choix (invariance par décalage).

    API à faire émerger (cible roadmap : `src/inference_lab/inference/sampling.py`)
    -----------------------------------------------------------------------------
        def greedy_next_token(logits: torch.Tensor) -> torch.Tensor: ...
            # (batch, vocab_size) float -> (batch,) long

    Indice : `logits.argmax(dim=-1)` renvoie déjà un tenseur `torch.long` de shape `(batch,)`.
    Pièges : `dim=-1` est obligatoire (sans `dim`, `argmax` aplatit tout le tenseur et rend un
    scalaire faux dès que `batch > 1`), et il ne faut PAS appliquer de softmax avant — c'est un
    calcul inutile qui ne change pas l'argmax.
    """

    pytest.skip("Roadmap TDD 3.6 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.inference.sampling import greedy_next_token

    # Arrange — construire `logits`, un tenseur float32 de shape (batch=1, vocab_size=16) dont
    #           les 16 valeurs sont toutes DISTINCTES (donc un maximum unique, sans ex aequo).
    #           Test déterministe : valeurs choisies à la main, ou `torch.manual_seed(0)`.

    # Act — demander le token greedy correspondant à ces logits, dans `token`.

    # Assert 1 — un id de token par élément du batch
    assert token.shape == (1,)
    assert token.dtype is torch.long
    assert 0 <= int(token[0]) < 16

    # Assert 2 — c'est bien l'index du logit maximum
    assert torch.equal(token, logits.argmax(dim=-1))

    # Assert 3 — strictement déterministe
    assert torch.equal(greedy_next_token(logits), greedy_next_token(logits))

    # Assert 4 — invariance par décalage : seul l'ordre des logits compte
    assert torch.equal(greedy_next_token(logits + 10.0), token)


@pytest.mark.tdd
@pytest.mark.model
def test_temperature_changes_probability_distribution():
    """Roadmap 3.7 — la temperature dilate ou contracte les écarts de logits avant le softmax.

    Objectif d'apprentissage
    ------------------------
    La temperature est le réglage de « créativité » d'un LLM, et c'est une simple division :
    `logits / T`. En dessous de 1, les écarts sont amplifiés, le softmax se concentre sur le
    meilleur candidat et le texte devient répétitif mais sûr ; au-dessus de 1, les écarts sont
    écrasés, la distribution s'aplatit vers l'uniforme et le modèle prend des risques. À la
    limite `T -> 0` on retrouve greedy, à `T -> +inf` un tirage uniforme. Elle s'applique
    TOUJOURS avant top-k et top-p, sinon on filtre sur une distribution qui n'est pas celle
    dans laquelle on tire.

    Schéma mental
    -------------
        logits (batch=1, vocab_size=16), écarts non nuls entre les valeurs

        T = 1.0 (froid=1)  ->  softmax(logits / 1) == softmax(logits)   INCHANGÉ
        T < 1   (ex. 0.5)  ->  écarts x2   -> probs.max() PLUS GRANDE   (concentration)
        T > 1   (ex. 2.0)  ->  écarts / 2  -> probs.max() PLUS PETITE   (aplatissement)

        toutes ces distributions somment à 1 et gardent le MÊME argmax

    Ce que ce test vérifie
    ----------------------
    1. `temperature = 1.0` laisse le softmax rigoureusement inchangé (la division par 1 est
       neutre : c'est le cas de référence à ne pas casser) ;
    2. une temperature < 1 concentre la distribution : la probabilité maximale augmente ;
    3. une temperature > 1 l'aplatit : la probabilité maximale diminue ;
    4. dans les trois cas on a toujours une distribution valide (somme 1) et le même token
       favori : la temperature change les probabilités, jamais le classement.

    API à faire émerger (cible roadmap : `src/inference_lab/inference/sampling.py`)
    -----------------------------------------------------------------------------
        def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor: ...
            # (batch, vocab_size) -> (batch, vocab_size), toujours des LOGITS

    Indice : `logits / temperature`, et rien de plus — la fonction rend des logits, pas des
    probabilités, pour rester composable avec top-k et top-p. Pièges : `temperature = 0`
    provoque une division par zéro (à traiter explicitement, ou à documenter comme équivalent
    à greedy) et une temperature négative inverserait le classement.
    """

    pytest.skip("Roadmap TDD 3.7 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.inference.sampling import apply_temperature

    # Arrange — construire `logits`, un tenseur float32 de shape (batch=1, vocab_size=16) aux
    #           valeurs toutes DISTINCTES (des écarts non nuls, sinon la temperature n'a rien
    #           à amplifier), et choisir deux réels : `cold` strictement entre 0 et 1, et
    #           `hot` strictement supérieur à 1.

    # Act — calculer les trois distributions issues du softmax (dim=-1) : `probs_base` sur les
    #       logits tels quels, `probs_cold` sur `apply_temperature(logits, cold)` et
    #       `probs_hot` sur `apply_temperature(logits, hot)`.

    # Assert 1 — temperature = 1 est le cas neutre : softmax inchangé
    torch.testing.assert_close(
        apply_temperature(logits, 1.0).softmax(dim=-1), probs_base, rtol=1e-5, atol=1e-6
    )

    # Assert 2 — temperature < 1 : la distribution se concentre
    assert float(probs_cold.max()) > float(probs_base.max())

    # Assert 3 — temperature > 1 : la distribution s'aplatit
    assert float(probs_hot.max()) < float(probs_base.max())

    # Assert 4 — ce sont toujours des distributions valides, et le favori ne change pas
    torch.testing.assert_close(probs_cold.sum(dim=-1), torch.ones(1), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(probs_hot.sum(dim=-1), torch.ones(1), rtol=1e-5, atol=1e-6)
    assert torch.equal(probs_cold.argmax(dim=-1), probs_hot.argmax(dim=-1))


@pytest.mark.tdd
@pytest.mark.model
def test_top_k_excludes_tokens_outside_k_highest_logits():
    """Roadmap 3.8 — top-k masque à `-inf`, ce qui donne une probabilité exactement nulle.

    Objectif d'apprentissage
    ------------------------
    Top-k coupe la queue de la distribution : on ne garde que les `k` meilleurs candidats.
    L'implémentation standard ne supprime pas les autres colonnes (cela casserait la
    correspondance index -> id de token) : elle met leur logit à `-inf`. Le softmax de `-inf`
    vaut exactement `0.0`, donc `torch.multinomial` ne pourra jamais les tirer. C'est la même
    mécanique que le masque causal de l'attention (section 2.4) : masquer, c'est écrire `-inf`
    AVANT le softmax, jamais mettre une probabilité à zéro après (ce qui donnerait une
    distribution non normalisée).

    Schéma mental
    -------------
        logits (batch=1, vocab_size=16), 16 valeurs distinctes, k = 4

        filtered : les 4 plus grands logits INTACTS, les 12 autres remplacés par -inf
             |
             v  softmax(dim=-1)
        probs : 4 valeurs > 0 qui somment à 1, et 12 valeurs EXACTEMENT 0.0

        16 - 4 = 12 exclus     4 candidats restants

    Ce que ce test vérifie
    ----------------------
    1. la shape est conservée `(1, 16)`, exactement 12 positions valent `-inf`, et `k = 16`
       (soit tout le vocabulaire) ne masque rien du tout ;
    2. après softmax, ces 12 positions ont une probabilité exactement `0.0`, et il reste
       exactement `k = 4` candidats de probabilité strictement positive ;
    3. la distribution filtrée reste normalisée (somme 1) : masquer avant le softmax
       renormalise automatiquement sur les survivants ;
    4. les `k` logits conservés sont inchangés, valeur par valeur : top-k sélectionne, il ne
       rééchelonne pas.

    API à faire émerger (cible roadmap : `src/inference_lab/inference/sampling.py`)
    -----------------------------------------------------------------------------
        def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor: ...
            # (batch, vocab_size) -> (batch, vocab_size), les exclus à -inf

    Indice : `torch.topk(logits, k, dim=-1)` donne valeurs et indices ; le seuil est la plus
    petite valeur retenue (`values[..., -1:]`), puis
    `logits.masked_fill(logits < seuil, -float("inf"))`. Pièges : utiliser `-float("inf")` et
    non un grand nombre négatif (sinon la probabilité n'est pas exactement 0), ne pas modifier
    `logits` en place, et gérer `k >= vocab_size` (rien à masquer).
    """

    pytest.skip("Roadmap TDD 3.8 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.inference.sampling import top_k_filter

    # Arrange — construire `logits`, un tenseur float32 de shape (batch=1, vocab_size=16) dont
    #           les 16 valeurs sont toutes DISTINCTES (aucun ex aequo, pour que « les k plus
    #           grands » soit sans ambiguïté), et fixer `k = 4`.

    # Act — filtrer `logits` avec top-k pour obtenir `filtered`, puis en déduire `probs`,
    #       le softmax de `filtered` sur la dernière dimension.

    # Assert 1 — shape conservée, 16 - 4 = 12 logits mis à -inf, et k = 16 ne masque rien
    assert filtered.shape == (1, 16)
    assert int(torch.isneginf(filtered).sum()) == 12
    assert int(torch.isneginf(top_k_filter(logits, 16)).sum()) == 0

    # Assert 2 — probabilité exactement nulle pour les exclus, exactement k candidats restants
    assert int((probs == 0.0).sum()) == 12
    assert int((probs > 0.0).sum()) == 4

    # Assert 3 — la distribution filtrée reste normalisée
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(1), rtol=1e-5, atol=1e-6)

    # Assert 4 — les logits conservés ne sont pas modifiés
    torch.testing.assert_close(
        filtered.topk(4, dim=-1).values, logits.topk(4, dim=-1).values, rtol=0.0, atol=0.0
    )


@pytest.mark.tdd
@pytest.mark.model
def test_top_p_limits_candidates_by_cumulative_probability():
    """Roadmap 3.9 — top-p garde le plus PETIT ensemble dont la masse cumulée atteint `p`.

    Objectif d'apprentissage
    ------------------------
    Top-k impose un nombre fixe de candidats, ce qui est arbitraire : quand le modèle est très
    sûr de lui, 4 candidats c'est 3 de trop ; quand il hésite, c'est trop peu. Top-p (nucleus
    sampling) rend ce nombre adaptatif : on trie les probabilités par ordre décroissant et on
    garde le plus court préfixe dont la somme atteint `p`. La taille du noyau varie donc à
    chaque token, ce qui explique pourquoi top-p est le réglage par défaut des API de
    génération, souvent combiné à top-k comme garde-fou.

    Schéma mental
    -------------
        probs triées (décroissant) : 0.50  0.30  0.12  0.05  ...     p = 0.9
        cumul                      : 0.50  0.80  0.92  0.97  ...
                                             ^ premier cumul >= 0.9 : on s'arrête ICI
        noyau gardé = 3 candidats (0.50, 0.30, 0.12), les 13 autres -> -inf

        minimalité : 0.92 - 0.12 = 0.80 < 0.9  -> retirer le dernier gardé repasse sous p

    Ce que ce test vérifie
    ----------------------
    1. la shape est conservée `(1, 16)`, au moins un candidat survit toujours (même si le
       plus probable dépasse déjà `p` à lui seul) et `p = 1.0` conserve tout le vocabulaire ;
    2. la masse cumulée des candidats gardés, mesurée dans la distribution d'ORIGINE,
       atteint `p = 0.9` ;
    3. l'ensemble est minimal : en retirer le plus petit candidat gardé repasse sous `p` ;
    4. les exclus ont une probabilité exactement `0.0` après filtrage, et tout candidat gardé
       est au moins aussi probable que n'importe quel exclu (on garde bien le HAUT de la
       distribution, pas un sous-ensemble quelconque).

    API à faire émerger (cible roadmap : `src/inference_lab/inference/sampling.py`)
    -----------------------------------------------------------------------------
        def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor: ...
            # (batch, vocab_size) -> (batch, vocab_size), les exclus à -inf

    Indice : `torch.sort(logits, descending=True, dim=-1)`, puis `softmax` et `cumsum` sur les
    valeurs triées ; on exclut les positions dont le cumul du PRÉDÉCESSEUR atteint déjà `p`
    (décalage de 1, sinon le premier candidat peut être supprimé quand sa probabilité dépasse
    `p`), puis `scatter` pour revenir à l'ordre des ids d'origine. Les tolérances de 1e-6 des
    asserts 2 et 3 absorbent l'arrondi float32 du `cumsum`.
    """

    pytest.skip("Roadmap TDD 3.9 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.inference.sampling import top_p_filter

    # Arrange — construire `logits`, un tenseur float32 de shape (batch=1, vocab_size=16) aux
    #           valeurs toutes DISTINCTES et suffisamment étalées pour que le noyau ne contienne
    #           pas les 16 tokens, et fixer `p = 0.9`.

    # Act — filtrer `logits` avec top-p pour obtenir `filtered`, puis calculer :
    #       `kept`, le masque booléen des positions non masquées (`filtered` n'y vaut pas -inf) ;
    #       `base_probs`, le softmax de `logits` d'origine sur la dernière dimension ;
    #       `kept_mass`, la somme de `base_probs` sur `kept` ;
    #       `smallest_kept`, le minimum de `base_probs` sur `kept`.

    # Assert 1 — shape conservée, noyau jamais vide, et p = 1.0 garde tout le vocabulaire
    assert filtered.shape == (1, 16)
    assert int(kept.sum()) >= 1
    assert int(torch.isneginf(top_p_filter(logits, 1.0)).sum()) == 0

    # Assert 2 — la masse gardée atteint p
    assert float(kept_mass) >= 0.9 - 1e-6

    # Assert 3 — ensemble minimal : sans son plus petit membre, on repasse sous p
    assert float(kept_mass) - float(smallest_kept) < 0.9 + 1e-6

    # Assert 4 — les exclus sont à probabilité nulle, et le noyau est bien le haut de la loi
    assert float(filtered.softmax(dim=-1)[~kept].sum()) == 0.0
    assert float(base_probs[kept].min()) >= float(base_probs[~kept].max())
