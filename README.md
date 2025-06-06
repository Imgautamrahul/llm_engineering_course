🧠 LLM Engineering with Ed Donner – Learning Repository

Welcome to my personal repository where I’m actively learning and implementing concepts from the **LLM Engineering course by Ed Donner**. This repo includes practical experiments, code implementations, and mini-projects related to large language models, prompt engineering, retrieval-augmented generation (RAG), and more.

---

 🚀 Repository Structure

 📁 `section5/`

A simple but complete implementation of a **brute-force RAG system**:

* Uses `SentenceTransformers` and `FAISS` for document retrieval
* Uses `Meta-LLaMA 3.2` as the reasoning engine
* Gradio UI for interactive Q\&A
* Grounded on local `.txt` files

 📁 `section4/`

* Converts **Python code to optimized C++** using a deployed HuggingFace endpoint
* Demonstrates stream-based interaction via `Gradio`

 📁 `section2/`

* Custom chatbots using `Gradio`, `LLaMA`, and prompt engineering
* Includes examples like:

  * **Samsung TV support assistant**
  * **Clothing store assistant**

---

 📚 Concepts Covered

* Prompt Engineering (`system`, `user`, `assistant` roles)
* Retrieval Augmented Generation (RAG)
* Embedding with `SentenceTransformers`
* Using open-source models via HuggingFace (`LLaMA 3.2`)
* Streamed output generation with `Gradio`
* Local deployment of LLMs in Jupyter Lab

---

 🛠 Tools & Technologies

* Python, Gradio, Transformers,Google Colab
* HuggingFace models (esp. LLaMA)
* Jupyter Lab for experimentation

---

 🔍 How to Use

1. Clone the repository
2. Install dependencies using `requirements.txt`
3. Place your data files (e.g., `.txt`) in appropriate folders (like `docs/`)
4. Run notebooks step-by-step in Jupyter Lab

---

 📌 Note

* You may need a **HuggingFace token** to use some gated models like `Meta-LLaMA 3.2`
* For local execution, ensure you have a **GPU-enabled environment** or use inference endpoints

---

 👨‍💻 Author

**Rahul Gautam** – Aspiring LLM Engineer | Tech Enthusiast

---


