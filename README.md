# ai_engineer_poc
Poc infer a model in multiple architectures

# Qwen Inference Engine — From PyTorch to GPU & Spyre

Projet pédagogique visant à construire progressivement un moteur d'inférence LLM autour de **Qwen**, en partant d'une implémentation PyTorch volontairement simple puis en ajoutant une à une les optimisations utilisées par les moteurs d'inférence modernes.

L'objectif principal n'est pas de battre vLLM en performances.

L'objectif est de comprendre la chaîne complète :

```text
Architecture LLM
      ↓
Tensors
      ↓
PyTorch operations
      ↓
Inference
      ↓
KV cache
      ↓
Prefill / Decode
      ↓
Kernels
      ↓
Memory
      ↓
Hardware
      ↓
Scheduling
      ↓
Performance
```

Le même modèle devra à terme pouvoir être exécuté sur différents backends :

```text
                        Qwen
                          │
                  Inference Engine
                          │
              ┌───────────┼───────────┐
              │           │           │
             CPU         CUDA       Spyre
```

---

# Modèle de départ

Le premier modèle utilisé est :

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Ce choix permet de travailler avec un vrai modèle pré-entraîné tout en conservant une taille suffisamment faible pour expérimenter rapidement.

L'architecture contient déjà plusieurs concepts importants pour les LLM modernes :

* Transformer decoder-only ;
* RMSNorm ;
* RoPE ;
* Grouped Query Attention ;
* SwiGLU ;
* KV cache pendant l'inférence.

Une fois notre implémentation validée, nous pourrons tester d'autres tailles de la même famille.

---

# Philosophie du projet

Nous suivons systématiquement :

```text
Comprendre
    ↓
Implémenter simplement
    ↓
Tester
    ↓
Comparer à une référence
    ↓
Mesurer
    ↓
Profiler
    ↓
Optimiser
    ↓
Remesurer
```

Nous évitons :

```text
copier une optimisation
       ↓
sans savoir ce qu'elle optimise
```

Chaque optimisation doit répondre à une question identifiable.

Exemple :

```text
Pourquoi le decode recalcule-t-il inutilement K et V ?

        ↓

KV Cache

        ↓

Quel gain mesurons-nous ?
```

---

# Références

Le projet utilise différentes références avec des rôles distincts.

## Hugging Face Transformers

Référence fonctionnelle.

Utilisé pour :

* télécharger les poids ;
* utiliser le tokenizer ;
* lire la configuration ;
* générer des résultats de référence ;
* comparer nos logits.

---

## nanochat

Référence pédagogique.

Utilisé pour étudier notamment :

* implémentation compacte d'un Transformer ;
* génération ;
* KV cache ;
* séparation prefill/decode ;
* benchmarking.

Le code de nanochat ne doit pas être copié aveuglément.

---

## vLLM

Référence pour les optimisations avancées du moteur :

* PagedAttention ;
* paged KV cache ;
* block allocator ;
* continuous batching ;
* scheduling ;
* prefix caching.

---

## Spyre

Backend matériel cible permettant de tester la portabilité du moteur au-delà des GPU CUDA.

---

# Architecture du repository

Architecture cible :

```text
qwen-engine/
│
├── model/
│   ├── config.py
│   ├── embeddings.py
│   ├── rmsnorm.py
│   ├── rope.py
│   ├── attention.py
│   ├── mlp.py
│   ├── block.py
│   └── qwen.py
│
├── inference/
│   ├── engine.py
│   ├── generation.py
│   ├── sampler.py
│   └── kv_cache.py
│
├── weights/
│   ├── hf_loader.py
│   └── mappings.py
│
├── backends/
│   ├── base.py
│   ├── cpu.py
│   ├── cuda.py
│   └── spyre.py
│
├── kernels/
│   ├── attention.py
│   ├── rope.py
│   ├── rmsnorm.py
│   └── matmul.py
│
├── benchmark/
│   ├── model.py
│   ├── prefill.py
│   ├── decode.py
│   └── kv_cache.py
│
├── tests/
│   ├── qwen/
│   │   ├── test_v0_hf_reference.py
│   │   ├── test_rmsnorm.py
│   │   ├── test_rope.py
│   │   ├── test_attention.py
│   │   ├── test_weights.py
│   │   ├── test_logits.py
│   │   ├── test_kv_cache.py
│   │   └── test_generation.py
│   └── README.md
│
├── docs/
│   ├── KILOCODE_SYSTEM_PROMPT.md
│   └── concepts/
│
├── scripts/
│   ├── generate.py
│   ├── compare_hf.py
│   └── benchmark.py
│
└── README.md
```

Cette structure est une cible.

Les dossiers et fichiers doivent être créés progressivement et non tous avant qu'ils deviennent nécessaires.

---

# Roadmap

# V0 — Hugging Face reference

## Objectif

Faire fonctionner le modèle officiel Qwen via Hugging Face et construire notre référence numérique.

```text
Prompt
   ↓
Tokenizer
   ↓
input_ids
   ↓
HF Qwen
   ↓
logits
   ↓
tokens
```

## Concepts

* tokenizer ;
* token IDs ;
* config ;
* checkpoint ;
* safetensors ;
* logits ;
* sampling / greedy decoding.

## Travail

Créer un script de référence capable de :

1. charger le tokenizer ;
2. charger le modèle ;
3. tokeniser un prompt fixe ;
4. exécuter un forward ;
5. récupérer les logits ;
6. générer quelques tokens.

## Validation

Conserver plusieurs résultats reproductibles qui serviront d'oracle pour notre modèle.

---

# V1 — Qwen forward from scratch

## Objectif

Réimplémenter le forward Qwen en PyTorch.

```text
input_ids
    ↓
Embedding
    ↓
Transformer Block × N
    ↓
Final RMSNorm
    ↓
LM Head
    ↓
Logits
```

## Composants

Implémenter progressivement :

```text
Config
  ↓
Embedding
  ↓
RMSNorm
  ↓
RoPE
  ↓
Q/K/V projections
  ↓
GQA
  ↓
Causal Attention
  ↓
SwiGLU
  ↓
Transformer Block
  ↓
QwenModel
```

## Règle

Aucune optimisation.

L'attention doit rester volontairement simple :

```text
Q × Kᵀ
   ↓
scale
   ↓
causal mask
   ↓
softmax
   ↓
× V
```

## Validation

Pour le même input :

```text
HF logits
    ≈
our logits
```

---

# V1.1 — Hugging Face weight loader

## Objectif

Charger les poids Qwen existants directement dans notre architecture.

```text
model.safetensors
       ↓
tensor names
       ↓
mapping
       ↓
our model
```

## Concepts

* `state_dict` ;
* safetensors ;
* weight mapping ;
* tensor shapes ;
* weight tying ;
* dtype.

## Validation

Tous les tensors doivent avoir :

* le nom attendu ;
* la shape attendue ;
* le dtype attendu.

Puis le forward complet doit reproduire les logits Hugging Face.

---

# V2 — Naive autoregressive generation

## Objectif

Créer notre première boucle de génération sans KV cache.

```text
Prompt
   ↓
Forward complet
   ↓
Token 1
   ↓
Prompt + Token 1
   ↓
Forward complet
   ↓
Token 2
   ↓
...
```

Pseudo-algorithme :

```python
for each generated token:

    logits = model(all_tokens)

    next_token = select(logits[:, -1])

    all_tokens = concatenate(all_tokens, next_token)
```

## Concepts

* autorégression ;
* causalité ;
* logits ;
* greedy decoding ;
* sampling ;
* coût du recalcul.

## Benchmark baseline

Mesurer :

* latency ;
* tokens/s ;
* mémoire.

Cette version devient notre baseline.

---

# V3 — KV Cache

## Objectif

Éviter de recalculer K et V pour les tokens précédents.

```text
Prompt
   ↓
K/V calculés
   ↓
KV Cache
   ↓
new token
   ↓
new K/V
   ↓
append cache
```

## Concepts

* Query ;
* Key ;
* Value ;
* temporal reuse ;
* cache par layer ;
* cache par sequence ;
* KV heads ;
* GQA ;
* mémoire par token.

## Questions auxquelles il faut savoir répondre

Pourquoi stocker K et V ?

Pourquoi ne stocke-t-on pas Q de la même manière ?

Pourquoi le cache augmente-t-il avec le contexte ?

Pourquoi GQA réduit-il fortement la taille du KV cache ?

## Validation

Les tokens produits doivent rester identiques à ceux de V2 en greedy decoding.

Comparer les performances V2 vs V3.

---

# V4 — Prefill and Decode

## Objectif

Séparer explicitement les deux workloads d'inférence.

```text
                       Request

                          │
              ┌───────────┴───────────┐
              │                       │
           PREFILL                  DECODE
              │                       │
         N prompt tokens          1 new token
              │                       │
          GEMM-like                GEMV-like
              │                       │
          KV creation              KV reuse
```

API cible :

```python
engine.prefill(tokens)
engine.decode()
```

## Concepts

### Prefill

* parallélisme sur les tokens ;
* création du KV cache ;
* compute intensity.

### Decode

* un token par étape ;
* lecture des poids ;
* lecture du KV cache ;
* memory bandwidth.

## Benchmark

Mesurer séparément :

```text
prefill latency
prefill tokens/s

decode latency
decode tokens/s
TPOT
TTFT
```

---

# V5 — Backend abstraction

## Objectif

Séparer les différences hardware de l'architecture Qwen.

```text
                       Model
                         │
                      PyTorch
                         │
               ┌─────────┼─────────┐
               │         │         │
              CPU       CUDA     Spyre
```

## Le backend peut gérer

* device ;
* dtype ;
* compilation ;
* synchronisation ;
* attention spécialisée ;
* KV cache allocation ;
* profiling ;
* kernels spécialisés.

## Le backend ne doit PAS abstraire

Chaque addition, multiplication ou opération PyTorch générique.

Éviter :

```python
backend.add()
backend.mul()
backend.matmul()
```

---

# V6 — Benchmark framework

## Objectif

Créer un protocole de comparaison reproductible.

Chaque benchmark doit enregistrer :

```text
model
backend
device
dtype
batch size
prompt length
output length
```

## Métriques

### Global

* load time ;
* total latency ;
* peak memory.

### Prefill

* TTFT ;
* input tokens/s ;
* prefill latency.

### Decode

* TPOT ;
* inter-token latency ;
* output tokens/s.

### Memory

* model memory ;
* KV cache memory ;
* peak device memory.

---

# V7 — Optimized Attention

## Baseline

```text
Naive PyTorch Attention
```

## Étape suivante

```text
PyTorch SDPA
```

Puis sur CUDA :

```text
FlashAttention
```

Puis si nécessaire :

```text
Spyre-specific attention
```

## Méthode

Pour chaque changement :

```text
correctness
    ↓
benchmark
    ↓
profiling
    ↓
analyse
```

---

# V8 — KV Memory Manager

## Objectif

Passer d'un cache simple à une vraie gestion mémoire.

Progression :

```text
KV cache contiguous
        ↓
preallocated KV cache
        ↓
memory pool
        ↓
fixed-size blocks
        ↓
block allocator
        ↓
block table
        ↓
paged KV cache
```

Puis étudier la relation avec PagedAttention.

## Concepts

* internal fragmentation ;
* external fragmentation ;
* logical blocks ;
* physical blocks ;
* block table ;
* allocation ;
* eviction.

---

# V9 — Batching & Scheduler

## Étape 1

```text
batch = 1
```

## Étape 2

```text
static batching
```

## Étape 3

```text
dynamic batching
```

## Étape 4

```text
continuous batching
```

## Architecture future

```text
Request A ──┐
Request B ──┼──► Scheduler
Request C ──┘        │
                     ↓
                token budget
                     │
              ┌──────┴──────┐
              │             │
           prefill        decode
              │             │
              └──────┬──────┘
                     │
                 accelerator
```

---

# V10 — Advanced Optimizations

Uniquement après maîtrise des étapes précédentes.

Sujets possibles :

```text
torch.compile
CUDA Graphs
kernel fusion
quantization
KV cache quantization
prefix caching
chunked prefill
continuous batching
speculative decoding
tensor parallelism
```

Chaque optimisation doit devenir une expérience mesurable.

---

# Tests

Les tests sont une partie centrale du projet.

Les tests sont organisés par famille de modèle afin de garder une séparation claire entre :

* la logique commune du moteur ;
* les spécificités d'une architecture donnée ;
* les références numériques utilisées pour valider cette architecture.

Structure actuelle :

```text
tests/
└── qwen/
    └── test_v0_hf_reference.py
```

Chaque dossier de modèle doit suivre la même progression que la roadmap.

Pour Qwen, la cible principale reste :

```text
Qwen/Qwen2.5-0.5B-Instruct
```

La première étape n'implémente aucun module custom. Elle vérifie uniquement que la référence Hugging Face fonctionne et produit des tenseurs de référence reproductibles.

```text
Prompt
   ↓
Tokenizer
   ↓
input_ids
   ↓
Hugging Face Qwen
   ↓
logits_ref
```

Shapes attendues en V0 :

```text
input_ids: [B, T]
attention_mask: [B, T]
logits_ref: [B, T, vocab_size]
next_token_logits: [B, vocab_size]
```

Cette organisation permet ensuite d'ajouter progressivement :

```text
tests/qwen/test_rmsnorm.py
tests/qwen/test_rope.py
tests/qwen/test_attention.py
tests/qwen/test_mlp.py
tests/qwen/test_weights.py
tests/qwen/test_logits.py
tests/qwen/test_kv_cache.py
tests/qwen/test_generation.py
```

Règle pédagogique : un test doit d'abord expliquer ce qu'il valide avant de chercher à couvrir tous les cas possibles.

Pour une nouvelle implémentation :

```text
unit test
   ↓
shape test
   ↓
numerical comparison
   ↓
full model comparison
```

Exemples :

```text
test_rmsnorm.py
test_rope.py
test_attention.py
test_mlp.py
test_weights.py
test_logits.py
test_kv_cache.py
test_generation.py
```

---

# Golden rule

Une version n'est pas validée uniquement parce qu'elle produit du texte.

Un LLM peut produire un texte plausible même lorsque l'implémentation contient une erreur.

Notre validation principale est :

```text
Reference implementation
          │
          ▼
        Tensor
          │
          │ compare
          │
          ▼
  Our implementation
```

---

# Definition of Done

Une étape est terminée lorsque :

* je peux expliquer le concept ;
* je connais les shapes principales ;
* le code est compréhensible ;
* les tests passent ;
* la sortie correspond à la référence ;
* les résultats sont reproductibles ;
* les performances sont mesurées lorsqu'elles sont pertinentes.

---

# Méthode d'étude

Pour chaque concept important, créer éventuellement une fiche dans :

```text
docs/concepts/
```

Format conseillé :

```markdown
# Concept

## Définition

## Pourquoi existe-t-il ?

## Mathématiques

## Tensor shapes

## Implémentation PyTorch

## Mémoire

## Prefill

## Decode

## Optimisations

## Benchmark / observations
```

Concepts prioritaires :

```text
RMSNorm
RoPE
GQA
SwiGLU
KV Cache
Prefill
Decode
Arithmetic Intensity
Paged KV Cache
Continuous Batching
```

---

# Benchmark philosophy

Ne jamais conclure :

> Cette version est plus optimisée.

sans mesurer.

Toujours comparer :

```text
Baseline
   ↓
Optimization
   ↓
Same workload
   ↓
Measurement
```

Puis répondre :

```text
Qu'est-ce qui s'est amélioré ?
Pourquoi ?
Quel coût supplémentaire ?
Quel bottleneck reste ?
```

---

# Long-term goal

À terme, le projet doit permettre l'expérience suivante :

```text
                SAME QWEN MODEL
                      │
             SAME MODEL WEIGHTS
                      │
                SAME PROMPT
                      │
          ┌───────────┴───────────┐
          │                       │
        CUDA                    Spyre
          │                       │
    optimized kernels       Spyre backend
          │                       │
          └───────────┬───────────┘
                      │
                   compare
                      │
       ┌──────────────┼──────────────┐
       │              │              │
      TTFT           TPOT          Memory
       │              │              │
       └──────────────┼──────────────┘
                      │
                 explain why
```

Le résultat attendu n'est pas uniquement un moteur fonctionnel.

Le résultat attendu est d'être capable de prendre un LLM et d'expliquer :

* comment ses poids sont représentés ;
* comment ses tenseurs circulent ;
* quelles opérations sont exécutées ;
* comment fonctionne son attention ;
* comment fonctionne son KV cache ;
* pourquoi prefill et decode ont des comportements différents ;
* où les données sont stockées ;
* quels kernels sont exécutés ;
* quel niveau de mémoire est sollicité ;
* quel est le bottleneck ;
* pourquoi une optimisation améliore ou non les performances ;
* comment adapter l'exécution à différents accélérateurs.
