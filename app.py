import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Hugging Face token from secrets
HF_TOKEN = st.secrets["HF_TOKEN"]
MODEL_NAME = "microsoft/DialoGPT-medium"

# Load the model and tokenizer (cached)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    return tokenizer, model

tokenizer, model = load_model()

# App title
st.title("🤖 Hugging Face Chatbot")
st.write("Ask me anything!")

# Initialize session state
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "past_inputs" not in st.session_state:
    st.session_state.past_inputs = []

# User input
user_input = st.text_input("You:", key="input")

if user_input:
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Append to conversation
    bot_input_ids = (
        torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)
        if st.session_state.chat_history_ids is not None
        else new_input_ids
    )

    output_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(output_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Save history
    st.session_state.past_inputs.append((user_input, response))
    st.session_state.chat_history_ids = output_ids

# Display conversation
for user_msg, bot_msg in st.session_state.past_inputs:
    st.markdown(f"*You:* {user_msg}")
    st.markdown(f"*Bot:* {bot_msg}")