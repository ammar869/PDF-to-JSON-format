import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from pdf2image import convert_from_path
import os

# --- CONFIGURATION ---
MODEL_PATH = "zai-org/GLM-OCR"
PDF_PATH = "Integration.pdf"

# WINDOWS USERS: If you get a "Poppler not found" error, un-comment the line below 
# and point it to your bin folder (e.g., r"C:\poppler\Library\bin")
POPPLER_PATH = None 
# POPPLER_PATH = r"C:\poppler\Library\bin" 

# ---------------------

def main():
    # 1. Device Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on: {device}...")

    # 2. Load Processor and Model
    # CRITICAL FIX: trust_remote_code=True must be on BOTH processor and model
    try:
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("Processor loaded successfully")
        
        # Check if processor has chat template
        print(f"Processor has chat template: {processor.chat_template is not None}")
        
        print("Loading model...")
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=None,  # Fix: Use None instead of device for CPU
            trust_remote_code=True
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Convert PDF to Image
    if not os.path.exists(PDF_PATH):
        print(f"Error: The file '{PDF_PATH}' was not found.")
        return

    print(f"Processing '{PDF_PATH}'...")
    try:
        # We pass poppler_path explicitly if the user defined it
        images = convert_from_path(PDF_PATH, poppler_path=POPPLER_PATH)
        image = images[0] # Take the first page
    except Exception as e:
        print(f"Error converting PDF. Is Poppler installed correctly?\nDetails: {e}")
        return

    # 4. Prepare the Prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Text Recognition:"}
            ],
        }
    ]

    # 5. Run Inference
    # We use torch.no_grad() to reduce memory usage significantly
    with torch.no_grad():
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        # Remove token_type_ids if present (fixes bugs with some custom models)
        inputs.pop("token_type_ids", None)

        print("Generating text (this may take a moment)...")
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=2048,
            do_sample=False # Deterministic output is usually better for OCR
        )
        
        output_text = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )

    print("\n" + "="*20 + " RESULT " + "="*20 + "\n")
    print(output_text)
    print("\n" + "="*48)

if __name__ == "__main__":
    main()