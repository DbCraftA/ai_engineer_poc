# Plan — Tests TDD pédagogiques (gabarit « exercice ») pour `tests/01_tensors/test_dtypes.py`

## État actuel

Étape pilote **déjà implémentée et validée** par l'utilisateur :

- `tests/01_tensors/test_dtypes.py:1-25` — docstring de module : cycle RED → GREEN → REFACTOR expliqué
  **une seule fois**, commande d'activation, convention Arrange/Act/Assert, renvoi vers
  `tests/ROADMAP.md`.
- `tests/01_tensors/test_dtypes.py:28-97` — `test_dtype_controls_bytes_per_element` réécrit
  intégralement (docstring pédagogique + corps complet). Il sert désormais d'**exemple corrigé de
  référence** : c'est le seul test du dépôt où le corps est entièrement écrit.
- `pyproject.toml:38` — `pythonpath = ["src"]` ajouté à `[tool.pytest.ini_options]`, sans quoi
  `inference_lab` n'est pas importable et l'étape GREEN est inatteignable.

Les deux autres tests du fichier sont encore au format généré (`expected = "..."` comparé à
lui-même, 12 `E501`).

## Décision nouvelle : les autres tests sont des exercices, pas des corrigés

Pour tous les tests **sauf le pilote**, on n'écrit dans le corps que :

1. la ligne `pytest.skip(...)` d'activation ;
2. le `from inference_lab.<module> import <api>` (c'est lui qui produit le RED) ;
3. des **commentaires** `# Arrange` et `# Act` qui décrivent le travail à faire et **nomment** les
   variables attendues ;
4. le **bloc `# Assert` complet et exécutable**, qui référence ces variables.

Si l'apprenant retire le `skip` sans faire l'Arrange, il obtient d'abord un `ModuleNotFoundError`
(module `src/` absent), puis un `NameError` : les deux échecs sont le signal explicite du travail
restant. Le contrat (les asserts) est donné, la construction et l'appel restent à écrire.

### Gabarit à appliquer

```python
@pytest.mark.tdd
def test_<nom>():
    """Roadmap <ID> — <phrase qui résume le concept>.

    Objectif d'apprentissage
    ------------------------
    <pourquoi ce concept compte dans un moteur d'inférence>

    Schéma mental
    -------------
    <mini-schéma chiffré : shapes, dtypes, résultat attendu>

    Ce que ce test vérifie
    ----------------------
    <liste numérotée, alignée sur les blocs Assert ci-dessous>

    API à faire émerger (cible roadmap : `src/inference_lab/<module>.py`)
    -------------------------------------------------------------------
        def <api>(...) -> <type>: ...

    Indice : <où trouver la réponse dans PyTorch, quel piège éviter>
    """

    pytest.skip("Roadmap TDD <ID> — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.<module> import <api>

    # Arrange — <quoi construire : nom(s) de variable(s), shape, dtype, propriété requise>

    # Act — <quel appel à l'API cible faire, en français>

    # Assert 1 — <ce que ce bloc démontre>
    assert ...
```

### Règles d'écriture

- ne jamais écrire de code Python dans les zones Arrange / Act : uniquement des commentaires ;
- les noms de variables attendus apparaissent entre backticks dans le commentaire `# Arrange` et
  sont utilisés tels quels dans les asserts ;
- shapes et dtypes peuvent être donnés (c'est le cadre de l'exercice) ; les **valeurs littérales
  d'entrée ne le sont pas**, on donne seulement la propriété requise ;
- les valeurs attendues restent écrites en dur dans les asserts (4, 2, 24, ...) ;
- 2 à 4 blocs `# Assert n — <ce qui est démontré>`, chacun mappé sur un point de la section
  « Ce que ce test vérifie » ;
- lignes ≤ 100 caractères (`ruff` E501, `line-length = 100`) ;
- ne pas toucher aux noms de tests ni au marker `@pytest.mark.tdd` (référencés par
  `tests/ROADMAP.md`).

## Tâches

### 1. `test_reduced_precision_changes_numerical_accuracy` (roadmap 1.8)

Cible : `src/inference_lab/tensors/dtypes.py`.

Docstring — contenu à couvrir :

- objectif : réduire la précision n'est pas gratuit ; le nombre de bits de mantisse fixe l'erreur
  d'arrondi. FP16 : 10 bits de mantisse, plage réduite. BF16 : 8 bits de mantisse, mais la plage
  d'exposant de FP32 (d'où son usage en inférence/entraînement) ;
- schéma mental : `x (float32) --.to(fp16)--> perte --.to(float32)--> x'`, et
  `erreur = max|x - x'|` ; illustrer avec une fraction non représentable en binaire (1/3) ;
- ce que le test vérifie : (1) aller-retour FP32→FP32 sans perte, (2) FP16 perd un peu mais reste
  dans une tolérance relative de 1e-3, (3) BF16 perd plus que FP16 à valeurs comparables,
  (4) l'aller-retour rend bien un tenseur FP32 ;
- API : `roundtrip(tensor, dtype) -> torch.Tensor` et `max_absolute_error(tensor, dtype) -> float` ;
- indice : `tensor.to(dtype).to(tensor.dtype)`, puis `(x - x_roundtrip).abs().max().item()` ;
  `torch.finfo(dtype)` donne `eps` et la plage.

Corps :

```python
    pytest.skip("Roadmap TDD 1.8 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.tensors.dtypes import max_absolute_error, roundtrip

    # Arrange — créer `x`, un petit tenseur 1D en `torch.float32` dont les valeurs ne sont PAS
    #           représentables exactement en binaire (des fractions, pas des puissances de deux),
    #           de magnitudes proches de 1 pour rester loin des limites de plage de FP16.

    # Act — faire l'aller-retour `x -> dtype -> float32` pour float32, float16 puis bfloat16,
    #       et mesurer l'erreur absolue maximale de chaque aller-retour.

    # Assert 1 — un aller-retour FP32 -> FP32 ne perd strictement rien
    assert max_absolute_error(x, torch.float32) == 0.0

    # Assert 2 — FP16 perd de l'information, mais reste dans ~1e-3 en relatif
    assert max_absolute_error(x, torch.float16) > 0.0
    torch.testing.assert_close(roundtrip(x, torch.float16), x, rtol=1e-3, atol=0.0)

    # Assert 3 — BF16 a 8 bits de mantisse contre 10 pour FP16 : il perd davantage
    assert max_absolute_error(x, torch.bfloat16) > max_absolute_error(x, torch.float16)

    # Assert 4 — l'aller-retour ramène bien le dtype d'origine
    assert roundtrip(x, torch.float16).dtype is torch.float32
```

### 2. `test_tensor_memory_equals_numel_times_element_size` (roadmap 1.9)

Cible : `src/inference_lab/tensors/memory.py`.

Docstring — contenu à couvrir :

- objectif : la formule `mémoire = numel x octets par élément`, réutilisée ensuite pour les poids
  du modèle, le KV cache et les activations ; c'est la base des calculateurs de la section 5 ;
- schéma mental : `(2, 3) float32 -> 6 x 4 = 24 octets` / `(2, 3) float16 -> 6 x 2 = 12 octets` ;
- ce que le test vérifie : (1) valeurs en octets écrites en dur, (2) cohérence avec
  `numel() * element_size()`, (3) cohérence avec l'allocation réelle
  `untyped_storage().nbytes()` pour un tenseur contigu fraîchement alloué, (4) à shape identique
  FP16 coûte deux fois moins que FP32 ;
- API : `tensor_memory_bytes(tensor: torch.Tensor) -> int` ;
- à mentionner sans l'asserter ici : une `view` partage le storage, donc elle ne coûte pas
  d'octets supplémentaires (c'est la section 1.3) — la fonction mesure la taille logique du
  tenseur, pas l'empreinte réelle du storage partagé.

Corps :

```python
    pytest.skip("Roadmap TDD 1.9 — supprimer cette ligne pour démarrer le cycle RED")

    from inference_lab.tensors.memory import tensor_memory_bytes

    # Arrange — créer `fp32` et `fp16`, deux tenseurs contigus fraîchement alloués de MÊME shape
    #           (2, 3), l'un en `torch.float32`, l'autre en `torch.float16`.

    # Act — demander à l'API la taille en octets de chacun des deux tenseurs.

    # Assert 1 — la formule appliquée à la main, en dur
    assert tensor_memory_bytes(fp32) == 24
    assert tensor_memory_bytes(fp16) == 12

    # Assert 2 — cohérence avec la formule générale, indépendante de la shape
    assert tensor_memory_bytes(fp32) == fp32.numel() * fp32.element_size()

    # Assert 3 — cohérence avec l'allocation réelle du storage
    assert tensor_memory_bytes(fp32) == fp32.untyped_storage().nbytes()

    # Assert 4 — à shape identique, FP16 coûte deux fois moins que FP32
    assert tensor_memory_bytes(fp32) == 2 * tensor_memory_bytes(fp16)
```

### 3. Nettoyage du fichier

- supprimer les lignes vides excédentaires en fin de fichier (`ruff format --check` les signale
  aujourd'hui, héritage du générateur) ;
- vérifier qu'il ne reste aucun `expected = "<nom du test>"` ni docstring générique dans le fichier.

### 4. Propagation (itération suivante, hors de ce lot)

Appliquer le gabarit « exercice » fichier par fichier, dans l'ordre de `tests/ROADMAP.md`
(`01_tensors` → `02_attention` → ...). Chaque fichier reçoit sa propre docstring de module
(cycle TDD + thème de la section). Ce n'est pas une substitution mécanique : chaque test demande un
schéma mental chiffré et une API cible spécifiques. Ne pas propager en masse.

## Validation

1. `pytest tests/01_tensors/test_dtypes.py -q` → 3 `skipped`, 0 erreur de collecte.
2. RED contrôlé test par test : retirer temporairement le `skip`, vérifier l'échec sur
   `ModuleNotFoundError: No module named 'inference_lab.tensors.<module>'` (l'import est placé avant
   les asserts, donc c'est bien lui qui échoue en premier), puis remettre le `skip`.
3. Cohérence des asserts vérifiée hors dépôt : implémenter les API dans un fichier jetable sous
   `/tmp`, le copier temporairement dans `src/`, lancer le test, **puis le supprimer**. Objectif :
   ne pas livrer un test impossible à passer (notamment l'assert BF16 > FP16, sensible au choix des
   valeurs de `x`). `src/` doit rester vide à la fin.
4. `uvx ruff check tests/01_tensors/test_dtypes.py` → 0 erreur (les 12 `E501` du template doivent
   disparaître) ; `uvx ruff format --check` sur ce fichier → clean.

## Hors périmètre

- Créer `src/inference_lab/tensors/dtypes.py` et `memory.py` : c'est l'exercice de l'utilisateur,
  `src/` reste inchangé.
- Modifier `tests/ROADMAP.md` (noms de tests et statuts inchangés).
- Les notebooks (`notebooks/01_tensors/...`) et les autres répertoires de tests.
