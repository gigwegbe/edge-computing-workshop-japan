# Load dependencies
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
from PIL import Image


# Load model and processor
model_id = "LiquidAI/LFM2-VL-3B" # LiquidAI/LFM2-VL-3B  # LiquidAI/LFM2-VL-1.6B
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="bfloat16",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


# ===== LOAD TEST IMAGE =====
image_path = "./detected/IMG_4826_detections.jpg"
image = Image.open(image_path)
if image.mode != "RGB":
    image = image.convert("RGB")



conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": (
                    "You are an expert toy inspector. Examine the image and identify each individual toy.\n\n"
                    "For every visible toy, assign a unique ID in the form: bear_1, bear_2, etc.\n\n"
                    "For each toy, determine:\n"
                    "- The dominant color(s)\n"
                    "- A concise visual description (type, outfit, accessories)\n\n"
                    "Return the result strictly as JSON (no extra text, no explanations) using the schema below:\n\n"
                    "{\n"
                    '  "objects": {\n'
                    '    "bear_1": {\n'
                    '      "colors": ["string"],\n'
                    '      "description": "string"\n'
                    "    },\n"
                    '    "bear_2": {\n'
                    '      "colors": ["string"],\n'
                    '      "description": "string"\n'
                    "    }\n"
                    "  }\n"
                    "}\n\n"
                    "Rules:\n"
                    "- Only include toys that are clearly visible.\n"
                    "- Do not guess unseen details.\n"
                    "- Base all descriptions strictly on visual evidence."
                )
            },
        ],
    },
]


# Generate Answer
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    tokenize=True,
).to(model.device)


outputs = model.generate(**inputs, max_new_tokens=64 * 3)
decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

# Extract only the assistant's reply
if "assistant" in decoded:
    response = decoded.split("assistant", 1)[1].strip()
else:
    response = decoded.strip()

print("\n--- Model Response ---")
print(response)