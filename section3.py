# minutes_of_meeting.py
# This script transcribes an audio file and generates meeting minutes using a language model.
from google.colab import drive
drive.mount('/content/drive')




import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline


whisper_model = "openai/whisper-small"

pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    chunk_length_s=30,
    device=0 if torch.cuda.is_available() else -1,
)


audio_path = "/content/drive/MyDrive/meeting_audio.mp3"


print("Transcribing audio...")
transcription = pipe(audio_path)["text"]
print("Transcription Done!")
print("\nTRANSCRIPT:\n", transcription)


from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # Use Hugging Face model name

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)


system_prompt = "You are an AI assistant that writes concise and professional meeting minutes."
user_prompt = f"Meeting Transcript:\n{transcription}\n\nCreate clear, structured meeting minutes."

prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

print("\nMEETING MINUTES:\n")
_ = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    do_sample=True,
    streamer=streamer
)
