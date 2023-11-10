import os
import warnings

import gradio as gr
from PIL import Image
from src.food import Food
from src import prompt_handler

warnings.filterwarnings("ignore")


def predict(image, hints):
    image_pil = Image.fromarray(image).convert("RGB")
    output = Food.generate(image_pil, hints)

    print(output)
    dict_out = prompt_handler.parse_dictionary_string(output)
    print(dict_out)


    isfood = dict_out.get('imageofFood', 'False')
    if not isfood:
        return 'Not food', 'Not food', {}
    menu_name = dict_out.get('menuName', 'Not found')
    description = dict_out.get('description', 'Not found')
    nutrients = dict_out.get('nutrients', 'Not found')
    nutrients_dict = {}
    if type(nutrients) is list:
        for nutrient in nutrients:
            name = nutrient['nutrientName']
            value = nutrient['nutritionalValue']
            unit = nutrient['unit']
            nutrients_dict[name] = f"{value} {unit}"

    return menu_name, description, nutrients_dict

inputs = [
    gr.Image(type="numpy", label='Image'),
    gr.Textbox(label="Hints", placeholder="Menu name, broccoli 3 pieces, steak 150g..."),
]

outputs = [
    gr.Textbox(label="Menu Name"),
    gr.Textbox(label="Description"),
    gr.JSON(label="Nutrients"),
]

examples = [

]

app = gr.Interface(fn=predict, inputs=inputs, outputs=outputs, examples=examples)
app.launch()
