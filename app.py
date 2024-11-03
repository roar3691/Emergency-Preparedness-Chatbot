import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import torch

# Set up the page configuration for Streamlit
st.set_page_config(page_title="Emergency Preparedness Chatbot", layout="centered")

# Path to the fine-tuned GPT-2 model and tokenizer (update this path as needed)
model_path = 'gpt2-finetuned-emergency/checkpoint-18'

# Detect if Apple Silicon MPS (Metal Performance Shaders) is available, otherwise use CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Cache the model and tokenizer to avoid reloading them on every interaction
@st.cache_resource
def load_model_and_tokenizer():
    """
    Load the fine-tuned GPT-2 model and tokenizer.
    This function is cached to avoid reloading the model and tokenizer on every interaction.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    return tokenizer, model

# Load the model and tokenizer once at the start
tokenizer, model = load_model_and_tokenizer()

# Create a text generation pipeline using the fine-tuned GPT-2 model
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Initialize session state to store chat history (this maintains conversation context)
if 'history' not in st.session_state:
    st.session_state.history = []

def generate_response(input_text):
    """
    Generate a response from the fine-tuned GPT-2 model based on user input and chat history.
    
    Args:
        input_text (str): The user's input message.
        
    Returns:
        None: The function appends the generated response to the session state history.
    """
    # Combine previous bot responses into context for chat awareness
    context = " ".join([item['bot'] for item in st.session_state.history]) + " " + input_text
    
    # Generate response from fine-tuned GPT-2 model using max_new_tokens instead of max_length
    response = generator(context, max_new_tokens=50, truncation=True, num_return_sequences=1)[0]['generated_text']
    
    # Append user input and bot response to history for continuity in conversation
    st.session_state.history.append({"user": input_text, "bot": response})

# Streamlit UI Layout: Title and instructions for the chatbot interface
st.title("Emergency Preparedness Chatbot")
st.write("Ask me anything about emergency preparedness!")

# Text input box for user queries
user_input = st.text_input("Your message:", "")

# When the user submits a message by clicking 'Send'
if st.button("Send"):
    if user_input:
        generate_response(user_input)

# Display chat history (conversation between user and bot)
if st.session_state.history:
    for chat in st.session_state.history:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")
