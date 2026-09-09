"""
V0 - Hugging Face reference for Qwen.

Milestone actuel : V0
Concept étudié : référence fonctionnelle Hugging Face
Objectif : charger Qwen/Qwen2.5-0.5B-Instruct, tokenizer un prompt fixe,
exécuter un forward et vérifier les shapes des tenseurs produits.
Critère de validation : logits_ref a la shape [B, T, vocab_size].

Ce fichier est volontairement un squelette pédagogique.
À compléter étape par étape avant d'implémenter notre propre Qwen.
"""

import pytest


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Explain in one sentence what a tensor is."


@pytest.mark.skip(reason="V0 à compléter manuellement : charger tokenizer + modèle Hugging Face puis vérifier les shapes.")
def test_qwen_hf_reference_forward_shapes():
    """
    À implémenter par étapes :

    1. Charger le tokenizer Hugging Face.
    2. Charger le modèle Hugging Face en mode eval.
    3. Tokenizer PROMPT.
    4. Vérifier input_ids.shape == [B, T].
    5. Exécuter un forward sans gradient.
    6. Vérifier logits_ref.shape == [B, T, vocab_size].

    Shapes attendues :
    - input_ids: [B, T]
    - attention_mask: [B, T]
    - logits_ref: [B, T, vocab_size]
    - next_token_logits: [B, vocab_size]
    """
    raise NotImplementedError("Implémentation volontairement laissée à l'étudiant.")
