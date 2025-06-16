from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import re
import unicodedata


PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "Eres un modelo experto en analizar discurso digital ofensivo y clasificarlo en una de tres categorías.\n"
    "Devuelve solo una línea de salida con la etiqueta exacta del siguiente formato:\n"
    "Respuesta = [fóbico / no fóbico / no relacionado]\n"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "Clasifica el siguiente mensaje: \"{}\"\n"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
)

def normalize(text):
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])
    return text

def extract_label(output_text):
    text = normalize(output_text)
    pattern = r"respuesta\s*=\s*\[\s*(f[oó]bico|no f[oó]bico|no relacionado)\s*\]"
    match = re.search(pattern, text)
    return match.group(1) if match else "desconocido"

def predict_llm(text, model_path):
    
    print(f"\n[LLM] Modelo: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    chat_prompt = [
        {"role": "system", "content": "Eres un modelo experto en analizar discurso digital ofensivo y clasificarlo en una de tres categorías."},
        {"role": "user", "content": (
            "Clasifica el siguiente mensaje como fóbico, no fóbico o no relacionado con respecto a la comunidad LGBTQ+. "
            "Devuelve solo una línea con la etiqueta exacta en este formato:\n"
            "Respuesta = [fóbico / no fóbico / no relacionado]\n\n"
            f"Mensaje: \"{text}\""
        )}
    ]
    # prompt = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)
    prompt = PROMPT_TEMPLATE.format(text)
    print(f"[LLM Prompt]\n{prompt}\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[LLM Output]\n{output_text}\n")

    label = extract_label(output_text)
    print(f"[Etiqueta Detectada]: {label}")

    label_map = {
        "no fóbico": [1, 0, 0],
        "fóbico": [0, 1, 0],
        "no relacionado": [0, 0, 1]
    }

    vector = label_map.get(label, [0.33, 0.33, 0.33])
    print(f"[Vector One-Hot]: {vector}")
    return label, vector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probar LLM con prompting de odio LGBTQ+")
    parser.add_argument("--text", type=str, required=True, help="Texto a evaluar")
    parser.add_argument("--model_path", type=str, required=True, help="Ruta del modelo local")
    args = parser.parse_args()

    predict_llm(args.text, args.model_path)
