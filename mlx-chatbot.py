import streamlit as st
from mlx_lm import load, generate
import time

st.set_page_config(page_title="Local Mistral-7B Chatbot", page_icon=":robot_face:", layout="wide")
st.markdown("<h1 style='text-align: center;'>Local Mistral-7B Chatbot</h1>", unsafe_allow_html=True)

# Cache the model loading to avoid reloading every time
@st.cache_resource
def load_model():
    model, tokenizer = load("mistralai/Mistral-7B-Instruct-v0.2")
    return model, tokenizer

model, tokenizer = load_model()

def generate_response(prompt: str, temp: float, max_tokens: int) -> str:
    response = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=max_tokens, temp=temp)
    return response

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.sidebar.title("Sidebar")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

if clear_button:
    st.session_state['messages'] = []

for message in st.session_state['messages']:
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.markdown(content)

# Chat input
prompt = st.chat_input("You:")
if prompt:
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    formatted_conversation = tokenizer.apply_chat_template(st.session_state['messages'], tokenize=False)
    tic = time.time()
    response = generate_response(formatted_conversation, temp=0.0, max_tokens=1000)

    st.session_state['messages'].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    tokens = tokenizer.tokenize(response)
    num_tokens_generated = len(tokens)
    generation_time = time.time() - tic
    generation_tps = num_tokens_generated / generation_time
    tokens = tokenizer.tokenize(formatted_conversation)
    num_tokens_total = len(tokens) + num_tokens_generated
    
    st.write(f"Number of tokens generated: {num_tokens_generated} --- Time: {generation_time:.1f} seconds --- TPS: {generation_tps:.1f}")
    st.write(f"Number of total tokens in conversation: {num_tokens_total}")
