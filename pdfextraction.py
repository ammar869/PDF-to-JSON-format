import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from pdf2image import convert_from_path
import os

# 1. Setup Model Path
MODEL_PATH = "zai-org/GLM-OCR"

# 2. Load Model and Processor
# We use device_map="cuda" if you have an NVIDIA GPU, otherwise "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on: {device}...")

processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map=device,
    trust_remote_code=True
)

# 3. Convert PDF to Image (Take the first page)
print("Processing PDF...")
pdf_path = "Integration.pdf" 
# Ensure poppler is in your PATH or specify poppler_path=r"C:\poppler\Library\bin"
images = convert_from_path(pdf_path) 
image = images[0] # Just take the first page for now

# 4. Prepare the Prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image}, # Pass the PIL image object directly
            {"type": "text", "text": "Text Recognition:"}
        ],
    }
]

# 5. Run Inference
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# Remove token_type_ids if present (fixes common bugs with some models)
inputs.pop("token_type_ids", None)

print("Generating text...")
generated_ids = model.generate(**inputs, max_new_tokens=2048)
output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print("\n--- Result ---\n")
print(output_text)