
# code to convert Python code to C++ using Meta's LLaMA-3 model

import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"


print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=True
)


def generate_cpp_code(python_code):
    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a high-performance software engineer AI. Translate Python code into efficient, readable, and optimized C++ code.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Convert the following Python code into high-performance C++ code:

{python_code}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generated_text = ""
    for output in model.generate(**inputs, streamer=streamer, max_new_tokens=512, temperature=0.5, top_p=0.9):
        pass 

    return generated_text


demo = gr.Interface(
    fn=generate_cpp_code,
    inputs=gr.Textbox(lines=15, placeholder="Paste your Python code here..."),
    outputs=gr.Textbox(label="Generated C++ Code"),
    title="🚀 Python to C++ Code Generator (LLaMA 3)",
    description="Using Meta's LLaMA-3 model hosted on HuggingFace to convert Python to high-performance C++"
)


demo.launch(debug=True)