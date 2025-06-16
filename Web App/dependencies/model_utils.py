
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = [
    "models/mdeberta_hp",
    "models/roberta_hp",
    "models/xlm_hp"
]

TRANSFORMER_MODELS = {}
for path in MODEL_PATHS:
    print(f"⚙️  Cargando {path}...")
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    TRANSFORMER_MODELS[path] = (model, tokenizer)

LLM_PATHS = [
    "models/llama3",
    "models/mistral_pt"
]

LLM_MODELS = {}
for path in LLM_PATHS:
    print(f"⚙️  Cargando LLM {path}...")
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    LLM_MODELS[path] = (model, tokenizer)

PROMPT_TEMPLATE = (
    "Eres un experto en lenguaje social y discurso digital. Analiza el siguiente mensaje "
    "y determina si expresa discurso de odio hacia la comunidad LGBTQ+. Justifica brevemente "
    "tu respuesta y luego proporciona una etiqueta final en este formato: "
    "Respuesta = [fóbico / no fóbico / no relacionado].\n\n"
    "[INST] \"{}\" [/INST]"
)

cached_models = {
    path: (
        AutoModelForSequenceClassification.from_pretrained(path).to(device).eval(),
        AutoTokenizer.from_pretrained(path)
    )
    for path in MODEL_PATHS
}

def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits.softmax(dim=1).cpu().numpy()[0]
    return scores
    
def predict_llm(text, model_path):
    print(f"\n[LLM] Modelo: {model_path}")
    model, tokenizer = LLM_MODELS[model_path]

    prompt = PROMPT_TEMPLATE.format(text)
    print(f"[LLM Prompt]\n{prompt}\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[LLM Output]\n{output_text}\n")

    label = "desconocido"
    if "respuesta = fóbico" in output_text.lower():
        label = "fóbico"
    elif "respuesta = no fóbico" in output_text.lower():
        label = "no fóbico"
    elif "respuesta = no relacionado" in output_text.lower():
        label = "no relacionado"

    print(f"[LLM Etiqueta Detectada]: {label}")

    label_map = {
        "no fóbico": [1, 0, 0],
        "fóbico": [0, 1, 0],
        "no relacionado": [0, 0, 1]
    }
    return label_map.get(label, [0.33, 0.33, 0.33])

def get_final_prediction_ensemble(text):
    print(f"\n{'='*70}")
    print(f"[ENSEMBLE] Texto de entrada:\n{text}")
    print(f"{'='*70}")

    # Transformers
    scores_list = []
    class_votes = []

    print("\n[Transformers]")
    for path in MODEL_PATHS:
        model, tokenizer = cached_models[path]
        scores = predict(text, model, tokenizer)
        scores_list.append(scores)
        predicted_class = scores.argmax()
        class_votes.append(predicted_class)
        print(f"  - Modelo: {path}")
        print(f"    → Clase predicha: {predicted_class}")
        print(f"    → Scores: {scores.tolist()}")

    from collections import Counter
    vote_count = Counter(class_votes)
    print(f"\n[Votación Transformers] {dict(vote_count)}")
    most_common = vote_count.most_common()

    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
        transformer_class = most_common[0][0]
        print(f"[Decision] Clase por mayoría: {transformer_class}")
    else:
        print("[Desempate] Empate en transformers → usando mdeberta como árbitro")
        mdeberta_model, mdeberta_tokenizer = cached_models[MODEL_PATHS[0]]
        scores = predict(text, mdeberta_model, mdeberta_tokenizer)
        transformer_class = scores.argmax()
        print(f"  → mdeberta desempate: {transformer_class}")

    transformer_avg = np.mean(scores_list, axis=0)
    print(f"\n[Promedio Transformers] {transformer_avg.tolist()}")

    # LLMs
    print("\n[LLMs]")
    llm_scores = []
    for path in LLM_PATHS:
        model, tokenizer = LLM_MODELS[path]
        prompt = PROMPT_TEMPLATE.format(text)
        print(f"\n  → Prompt para {path}:\n{prompt}\n")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0
            )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  → Output {path}:\n{output_text}\n")

        label = "desconocido"
        if "respuesta = fóbico" in output_text.lower():
            label = "fóbico"
        elif "respuesta = no fóbico" in output_text.lower():
            label = "no fóbico"
        elif "respuesta = no relacionado" in output_text.lower():
            label = "no relacionado"

        label_map = {
            "no fóbico": [1, 0, 0],
            "fóbico": [0, 1, 0],
            "no relacionado": [0, 0, 1]
        }
        vector = label_map.get(label, [0.33, 0.33, 0.33])
        print(f"  → Etiqueta: {label} → Vector: {vector}")
        llm_scores.append(vector)

    llm_avg = np.mean(llm_scores, axis=0)
    print(f"\n[Promedio LLMs] {llm_avg.tolist()}")

    # Combinación
    alpha = 0.5
    final_scores = alpha * transformer_avg + (1 - alpha) * llm_avg
    final_class = final_scores.argmax()

    print(f"\n[Fusión final] alpha = {alpha}")
    print(f"  → Scores combinados: {final_scores.tolist()}")
    print(f"  → Clase final: {final_class}")
    print(f"{'='*70}\n")

    return int(final_class), final_scores