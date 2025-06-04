#  This script uses the Ollama API to stream responses from a model
#  for a technical tutor application. It sends a prompt to the model and streams the response back in real-time.
import requests
import json

def stream_from_ollama(prompt, model="llama3.2"):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }

    print("\n🤖 Generating...\n")
    with requests.post(url, headers=headers, data=json.dumps(data), stream=True) as response:
        for line in response.iter_lines():
            if line:
                line_data = json.loads(line.decode("utf-8"))
                if "response" in line_data:
                    print(line_data["response"], end="", flush=True)

def technical_tutor_stream(text, question=None, model="llama3.2"):
    prompt = f"""You are a helpful technical tutor. Read the following content and explain it in simple steps:\n\n{text}\n"""
    if question:
        prompt += f"\nNow, answer this question clearly and simply:\n{question}\n"
    
    stream_from_ollama(prompt, model)
text = """
In machine learning, backpropagation is an algorithm used to train neural networks. It works by calculating the gradient of the loss function with respect to each weight by the chain rule, propagating the error backward from the output layer to the input layer. This allows the network to adjust the weights to minimize the error.
"""

question = "What is backpropagation and why is it important?"

technical_tutor_stream(text, question, model="llama3.2")
