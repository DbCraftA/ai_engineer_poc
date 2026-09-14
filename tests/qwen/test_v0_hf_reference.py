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


@pytest.mark.hf
@pytest.mark.slow
@pytest.mark.skip(reason="Heavy V0 HF reference: enable manually after the lightweight foundation is stable.")
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
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    messages = [{ 'role': "user", "content": PROMPT}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs,max_new_tokens=40)
    result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    print(result)
    model.forward
