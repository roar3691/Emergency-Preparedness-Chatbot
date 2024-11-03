import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import torch

# Set up the page configuration
st.set_page_config(page_title="Emergency Preparedness Chatbot", layout="centered")

# Load the fine-tuned GPT-2 model and tokenizer
model_path = '/Users/yanalaraghuvamshireddy/emergency_chatbot/gpt2-finetuned-emergency/checkpoint-18'
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the fine-tuned model and tokenizer once at the start
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Create a text generation pipeline using the fine-tuned model
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Initialize session state to store chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Function to generate responses with chat history (chat awareness)
def generate_response(input_text):
    # Combine previous bot responses into context for chat awareness
    context = " ".join([item['bot'] for item in st.session_state.history]) + " " + input_text
    
    # Generate response from fine-tuned GPT-2 model using max_new_tokens instead of max_length
    response = generator(context, max_new_tokens=50, truncation=True, num_return_sequences=1)[0]['generated_text']
    
    # Append user input and bot response to history
    st.session_state.history.append({"user": input_text, "bot": response})

# Streamlit UI Layout
st.title("Emergency Preparedness Chatbot")
st.write("Ask me anything about emergency preparedness!")

# Text input box for user queries
user_input = st.text_input("Your message:", "")

# When the user submits a message
if st.button("Send"):
    if user_input:
        generate_response(user_input)

# Display chat history
if st.session_state.history:
    for chat in st.session_state.history:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")