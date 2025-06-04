# clothing ai assistant
# This code creates a Gradio app that allows users to chat with an AI assistant for a clothing store.
import gradio as gr
import ollama

# --- System Prompt ---
system_prompt = """
You are a helpful and friendly AI assistant for a clothing store.
You help customers find information about products, order status, returns, and sizing.
Always be polite, concise, and supportive in your responses.
"""

# --- Chat History ---
history = [] 

# --- Chat Function ---
def chat_with_llama(message, history_dict):
    history_dict.append({"role": "user", "content": message})
    messages = [{"role": "system", "content": system_prompt}] + history_dict

    stream = ollama.chat(
        model="llama3",
        messages=messages,
        stream=True
    )

    full_response = ""
    for chunk in stream:
        if 'content' in chunk['message']:
            content = chunk['message']['content']
            full_response += content
            yield full_response, history_dict  # Streaming each part

    history_dict.append({"role": "assistant", "content": full_response})

#
with gr.Blocks() as demo:
    gr.Markdown("## Clothing Store AI Assistant (Powered by LLaMA 3.2)")

    chatbot = gr.Chatbot(label="Your Assistant", type='messages')
    msg = gr.Textbox(placeholder="Ask me anything about our clothing store...", show_label=False)
    state = gr.State([])  # Maintains history

    def user_submit(user_message, chat_history):
        return gr.update(value="", interactive=False), chat_history + [("user", user_message)]

    msg.submit(user_submit, [msg, state], [msg, chatbot], queue=False).then(
        chat_with_llama, [msg, state], [chatbot, state]
    )

# --- Launch ---
demo.launch(debug=True)
