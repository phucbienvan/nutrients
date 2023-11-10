import src.prompt_handler as prompt_handler
import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch


class Food:
    def __init__(self,):
        print("1")

    def generate(pil_image, hints= "there is a bowl of watermelon sitting on a computer desk"):
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xxl")
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        prompt = prompt_handler.get_prompt(hints)

        inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)

        outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
        )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        print(generated_text)

