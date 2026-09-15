"""Section 11.4 / 11.5 — backend d'attention interchangeable et garde-fou d'équivalence.

Comment travailler les tests de ce fichier (cycle TDD)
------------------------------------------------------
Chaque test est une *spécification exécutable* : il décrit le comportement attendu
AVANT que le code de `src/` n'existe.

1. RED      : supprimer la ligne `pytest.skip(...)` du test, puis lancer
              `pytest tests/11_custom_llm/test_attention_backend.py -k <nom_du_test>`.
              Le test DOIT échouer : le module cible dans `src/` n'existe pas encore.
2. GREEN    : écrire le minimum de code dans le module `src/` indiqué par le test,
              juste assez pour faire passer les assertions, rien de plus.
3. REFACTOR : nettoyer ce code sans changer le comportement ; le test reste vert et
              devient le filet de sécurité.

Lecture d'un test : `Arrange` prépare les données, `Act` appelle l'API cible,
`Assert` compare au comportement attendu. Les valeurs attendues sont écrites en dur
("naive" / "sdpa", logits (1, 6, 32), `ValueError` sur un backend inconnu, tolérance fp32
rtol=1e-5 / atol=1e-6) : c'est volontaire. Un test doit énoncer la vérité attendue, pas la
recalculer avec la même formule que le code testé.

Dimensions jouets constantes de toute la section 11 : vocab=32, hidden=16, num_layers=2,
num_heads=4, num_kv_heads=2 (GQA), head_dim=4, intermediate=32, seq=6, batch=1.
Différence essentielle avec la section 11.3 : changer de normalisation change le modèle,
changer de backend d'attention ne DOIT rien changer. `naive` et `sdpa` implémentent la même
mathématique (sections 2.x et 8.1) avec un ordre de réduction différent. Le test 11.5 est le
plus important de la section : c'est lui qui autorise toute optimisation ultérieure
(SDPA, kernels Triton de la section 9, `torch.compile`) en prouvant qu'elle ne modifie pas
les sorties du modèle.

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
def test_custom_llm_can_select_naive_or_sdpa_attention():
    """Roadmap 11.4 — le chemin d'exécution de l'attention est un paramètre, pas un choix figé.

    Objectif d'apprentissage
    ------------------------
    Un backend d'attention est une décision d'IMPLÉMENTATION : `naive` matérialise la matrice
    de scores (seq, seq) en quatre appels lisibles, `sdpa` appelle
    `torch.nn.functional.scaled_dot_product_attention` en un seul kernel fusionné, sans jamais
    allouer cette matrice (section 8.1). Rendre ce choix configurable est ce qui permet de
    mesurer honnêtement un gain : même modèle, mêmes poids, mêmes entrées, seul le chemin de
    calcul change. C'est le mécanisme derrière `attn_implementation="eager" | "sdpa" |
    "flash_attention_2"` de Hugging Face. Ici encore, un backend inconnu doit lever une erreur
    immédiatement : un repli silencieux sur `naive` transformerait un benchmark en mensonge —
    on croirait mesurer SDPA alors qu'on mesure le chemin lent.

    Schéma mental
    -------------
        config.attention_backend = "naive"  -> scores (1, 4, 6, 6) alloués, softmax, @ V
        config.attention_backend = "sdpa"   -> un seul kernel, aucune matrice (6, 6) allouée
        config.attention_backend = "flash"  -> ValueError (pas implémenté ici)

        les deux chemins : q (1, 4, 6, 4), k et v (1, 2, 6, 4) répétés sur 4 têtes
                           -> sortie (1, 4, 6, 4) -> logits (1, 6, 32)

        le backend est LISIBLE sur le modèle instancié : model.config.attention_backend

    Ce que ce test vérifie
    ----------------------
    1. le backend déclaré est lisible sur le modèle instancié, et les deux valeurs supportées
       sont acceptées ;
    2. le backend ne change ni la structure ni le nombre de poids : les deux modèles ont
       exactement les mêmes noms de paramètres dans leur `state_dict()` ;
    3. les deux modèles produisent des logits (1, 6, 32) en float32, finis ;
    4. un backend inconnu lève une `ValueError`.

    API à faire émerger (cible roadmap : « custom model », cibles proposées
    `src/inference_lab/models/custom_llm/config.py` et
    `src/inference_lab/models/custom_llm/model.py`)
    ---------------------------------------------------------------------
        class CustomLLMConfig:
            attention_backend: str = "naive"    # "naive" | "sdpa", validé à la création

        class CustomLLM(torch.nn.Module):
            def forward(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    Indice : le backend ne doit toucher QUE le cœur du calcul d'attention, jamais les
    projections : `q_proj`, `k_proj`, `v_proj`, `o_proj` et la répétition des têtes KV sont
    communes aux deux chemins. C'est cette factorisation qui rend l'assert 2 vrai, et le test
    11.5 possible. Valide `attention_backend` à la construction de la configuration, comme
    `norm_type` en 11.3. Piège : oublier le masque causal du côté `sdpa` — les deux chemins
    doivent utiliser la même règle de masquage (`is_causal=True` en prefill), sinon 11.5
    échouera.
    """

    pytest.skip("Roadmap TDD 11.4 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.models.custom_llm.config import CustomLLMConfig
    from inference_lab.models.custom_llm.model import CustomLLM

    # Arrange — dimensions jouets de la section : vocab=32, hidden=16, num_layers=2,
    #           num_heads=4, num_kv_heads=2, head_dim=4, intermediate=32, seq=6, batch=1.
    #           `model_naive` : un `CustomLLM` construit avec `attention_backend="naive"`.
    #           `model_sdpa` : le MÊME jeu de dimensions avec `attention_backend="sdpa"`.
    #           Les deux modèles sont en `torch.float32`, sur CPU, en `eval()`.
    #           `input_ids` : tenseur d'entiers (1, 6) à valeurs dans [0, 32), déterministe
    #           (seed fixée), partagé par les deux modèles.

    # Act — calculer `logits_naive = model_naive(input_ids)` et
    #       `logits_sdpa = model_sdpa(input_ids)`.

    # Assert 1 — le backend est lisible sur le modèle instancié
    assert isinstance(model_naive, CustomLLM)
    assert model_naive.config.attention_backend == "naive"
    assert model_sdpa.config.attention_backend == "sdpa"

    # Assert 2 — même structure de poids des deux côtés : seul le calcul change
    assert set(model_naive.state_dict()) == set(model_sdpa.state_dict())
    assert len(model_naive.blocks) == len(model_sdpa.blocks) == 2

    # Assert 3 — même contrat de sortie
    assert logits_naive.shape == (1, 6, 32)
    assert logits_sdpa.shape == (1, 6, 32)
    assert logits_sdpa.dtype is torch.float32
    assert bool(logits_sdpa.isfinite().all())

    # Assert 4 — backend inconnu : échec immédiat
    with pytest.raises(ValueError):
        CustomLLMConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            head_dim=4,
            intermediate_size=32,
            attention_backend="flash",
        )


@pytest.mark.tdd
@pytest.mark.model
def test_equivalent_attention_backends_produce_close_outputs():
    """Roadmap 11.5 — même mathématique, ordre de réduction différent : sorties équivalentes.

    Objectif d'apprentissage
    ------------------------
    C'est le test le plus important de la section, et le garde-fou de TOUTE optimisation
    ultérieure du dépôt. Le raisonnement : `naive` et `sdpa` calculent la même fonction
    mathématique, `softmax(Q K^T / sqrt(d)) V`. Mais SDPA regroupe les opérations en un kernel
    fusionné, traite les scores par blocs et emploie un softmax en ligne : les additions
    flottantes ne sont donc pas faites dans le même ordre. Or l'addition flottante n'est pas
    associative — (a+b)+c ne vaut pas exactement a+(b+c). L'écart attendu en fp32 est de
    l'ordre de 1e-7 : ce n'est PAS un bug, c'est le bruit d'arrondi. D'où deux règles à
    retenir : on compare avec `torch.testing.assert_close` et une tolérance (jamais
    `torch.equal`), et l'écart doit rester au niveau du bruit. Si cet écart grimpe à 1e-2,
    ce n'est plus de l'arrondi : c'est un masque oublié, une échelle différente, ou des têtes
    KV répétées dans le mauvais ordre. Chaque optimisation future — kernels Triton
    (section 9), `torch.compile`, quantification — devra passer par ce même test.

    Schéma mental
    -------------
        model_naive (backend "naive")            model_sdpa (backend "sdpa")
            |                                        |
            +---- load_state_dict(mêmes poids) ------+   <- indispensable
            |                                        |
          eval()                                   eval()                <- indispensable
            |                                        |
        logits (1, 6, 32)                        logits (1, 6, 32)

        écart mesuré en fp32 : ~2e-7  ->  passe avec rtol=1e-5, atol=1e-6
        écart d'un vrai bug (masque oublié) : ~1e-1  ->  échoue, et c'est le but

    Ce que ce test vérifie
    ----------------------
    1. les deux modèles partent bien du MÊME état : même jeu de clés de `state_dict()` et
       poids identiques bit à bit après `load_state_dict` ;
    2. les logits coïncident à la tolérance fp32 rtol=1e-5 / atol=1e-6, sur toute la sortie
       (1, 6, 32) ;
    3. la conclusion pratique : le token prédit à chaque position est le même des deux côtés
       (`argmax` identique) — le backend n'a aucun effet observable sur le modèle ;
    4. l'équivalence tient aussi position par position sur le dernier token, celui qui sert au
       decode, et aucune sortie ne contient de valeur non finie.

    API à faire émerger (cible roadmap : « custom model », cibles proposées
    `src/inference_lab/models/custom_llm/config.py` et
    `src/inference_lab/models/custom_llm/model.py`)
    ---------------------------------------------------------------------
        class CustomLLM(torch.nn.Module):
            def forward(self, input_ids: torch.Tensor) -> torch.Tensor: ...
            # `state_dict()` / `load_state_dict()` viennent de torch.nn.Module

    Indice : deux précautions rendent ce test fiable, et leur oubli est la première cause
    d'échec. D'abord charger les MÊMES poids : deux instanciations successives tirent des
    initialisations différentes, donc `model_sdpa.load_state_dict(model_naive.state_dict())`
    est obligatoire. Ensuite `eval()` sur les deux modèles, et le calcul sous
    `torch.no_grad()` : tout dropout ou statistique en mode entraînement introduirait un écart
    aléatoire qui n'a rien à voir avec l'arrondi. Reste le vrai piège de fond : les deux
    chemins doivent appliquer EXACTEMENT le même masque causal et la même échelle
    `1/sqrt(head_dim)` — SDPA applique déjà cette échelle implicitement, ne la mets pas deux
    fois.
    """

    pytest.skip("Roadmap TDD 11.5 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.models.custom_llm.config import CustomLLMConfig
    from inference_lab.models.custom_llm.model import CustomLLM

    # Arrange — dimensions jouets de la section : vocab=32, hidden=16, num_layers=2,
    #           num_heads=4, num_kv_heads=2, head_dim=4, intermediate=32, seq=6, batch=1,
    #           `torch.float32` sur CPU (c'est la précision qui rend la tolérance 1e-6 tenable).
    #           `model_naive` : un `CustomLLM` avec `attention_backend="naive"`, poids
    #           déterministes (seed fixée).
    #           `model_sdpa` : un `CustomLLM` avec `attention_backend="sdpa"` et les MÊMES
    #           dimensions, dans lequel on charge les poids de `model_naive` via
    #           `model_sdpa.load_state_dict(model_naive.state_dict())` — sans ce chargement, les
    #           deux modèles n'ont pas les mêmes poids et la comparaison ne veut rien dire.
    #           Les deux modèles sont ensuite mis en `eval()`.
    #           `input_ids` : tenseur d'entiers (1, 6) à valeurs dans [0, 32), déterministe,
    #           STRICTEMENT le même tenseur pour les deux modèles.

    # Act — sous `torch.no_grad()`, calculer `logits_naive = model_naive(input_ids)` et
    #       `logits_sdpa = model_sdpa(input_ids)`.

    # Assert 1 — même point de départ : les poids sont identiques bit à bit
    assert isinstance(model_naive, CustomLLM)
    assert isinstance(model_sdpa.config, CustomLLMConfig)
    assert set(model_naive.state_dict()) == set(model_sdpa.state_dict())
    assert torch.equal(
        model_naive.blocks[0].attention.k_proj.weight,
        model_sdpa.blocks[0].attention.k_proj.weight,
    )
    assert torch.equal(model_naive.lm_head.weight, model_sdpa.lm_head.weight)

    # Assert 2 — même mathématique, ordre de réduction différent : égalité à ~1e-7 près
    assert logits_naive.shape == (1, 6, 32)
    torch.testing.assert_close(logits_naive, logits_sdpa, rtol=1e-5, atol=1e-6)

    # Assert 3 — conséquence observable : le modèle prédit exactement les mêmes tokens
    assert torch.equal(logits_naive.argmax(dim=-1), logits_sdpa.argmax(dim=-1))

    # Assert 4 — l'équivalence vaut aussi pour le dernier token, celui utilisé en decode
    torch.testing.assert_close(logits_naive[:, -1, :], logits_sdpa[:, -1, :], rtol=1e-5, atol=1e-6)
    assert bool(logits_naive.isfinite().all())
    assert bool(logits_sdpa.isfinite().all())
