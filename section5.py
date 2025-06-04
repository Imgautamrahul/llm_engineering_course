# effectively using RAG with LLaMA for insurance tech queries
# This script uses LLaMA-3 to answer insurance-related questions by retrieving relevant documents and generating responses.
import os
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import gradio as gr


llama_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
print("Loading LLaMA model...")
tokenizer = AutoTokenizer.from_pretrained(llama_model_id, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    llama_model_id, torch_dtype=torch.float16, device_map="auto", use_auth_token=True
)


embed_model = SentenceTransformer("all-MiniLM-L6-v2")


doc_folder = "rahul_tax.docs"  
texts, ids = [], []

for filename in os.listdir(doc_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(doc_folder, filename), "r", encoding="utf-8") as f:
            content = f.read()
            texts.append(content)
            ids.append(filename)

print(f"Loaded {len(texts)} documents from {doc_folder}.")


embeddings = embed_model.encode(texts, convert_to_tensor=True).cpu().detach().numpy()
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


def rag_query(user_question):
  
    question_embedding = embed_model.encode([user_question])[0].reshape(1, -1)
    D, I = index.search(question_embedding, k=3)


    retrieved_docs = [texts[i] for i in I[0]]
    context = "\n---\n".join(retrieved_docs)

   
    system_prompt = "You are an expert assistant for an insurance tech company. Be accurate and helpful."
    user_prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {user_question}"

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.3,
        top_p=0.95
    )
    return tokenizer.decode(output[0], skip_special_tokens=True).split("assistant")[-1].strip()


demo = gr.Interface(
    fn=rag_query,
    inputs=gr.Textbox(label="Ask your insurance-related question"),
    outputs=gr.Textbox(label="Answer"),
    title=" Expert Knowledge Worker for Insurellm",
    description="Ask questions about policies, claims, or any internal insurance content."
)


demo.launch(debug=True)
