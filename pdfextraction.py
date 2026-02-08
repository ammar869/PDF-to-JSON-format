from pdf2image import convert_from_path
from transformers import GLMProcessor, AutoModelForImageTextToText
import torch
import json

MODEL_PATH = "zai-org/GLM-OCR"

print("Loading GLM-OCR model...")
processor = GLMProcessor.from_pretrained(MODEL_PATH)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)

print("Converting PDF to images...")
pages = convert_from_path("input.pdf", dpi=300)

results = {}

for i, page in enumerate(pages):
    print(f"OCR page {i}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": page},  # Pass PIL image directly
                {
                    "type": "text",
                    "text": """
Extract MCQs from this document.

Return STRICT JSON format:

{
 "question_number":{
   "body":"",
   "a":"",
   "b":"",
   "c":"",
   "d":"",
   "correct_option":""
 }
}

Output ONLY JSON.
"""
                }
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    inputs.pop("token_type_ids", None)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=4096
    )

    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    try:
        parsed = json.loads(output_text)
        results.update(parsed)
    except:
        print("⚠ JSON parsing failed on page", i)

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("DONE. output.json created.")
