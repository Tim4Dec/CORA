import streamlit as st
import json
import os, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import PeftModel


fs = open('chat_log.txt', 'a+')
fs.write(os.path.abspath(__file__))

def on_btn_click():
    del st.session_state.messages

def set_config():
    base_config = {"model_name": ""}
    model_config = {'top_k': '', 'top_p': '', 'temperature': '', 'max_length': '', 'do_sample': ""}

    with st.sidebar:
        model_name = st.radio(
            ["chatglm3-6b", "Llama-2-7b-chat-hf", "Meta-Llama-3-8B", "Qwen"],
            index=0
        )
        base_config['model_name'] = model_name

        sample = st.radio("Do Sample", ('True', 'False'))
        max_length = st.slider("Max Length", min_value=64, max_value=2048, value=512)
        top_p = st.slider(
            'Top P', 0.0, 1.0, 0.7, step=0.01
        )
        temperature = st.slider(
            'Temperature', 0.0, 2.0, 0.95, step=0.01
        )
        st.button("Clear Chat History", on_click=on_btn_click)

    model_config['top_p'] = top_p
    model_config['do_sample'] = sample
    model_config['max_length'] = max_length
    model_config['temperature'] = temperature
    return base_config, model_config

def set_input_format(model_name):
    if model_name == "Llama-2-7b-chat-hf" :
        input_format = "<reserved_106>{{query}}<reserved_107>"
    elif model_name == "Qwen":
        input_format = """
        <|im_start|>system 
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        {{query}}<|im_end|>
        <|im_start|>assistant"""
    elif model_name == "Meta-Llama-3-8B":
        input_format = """{{query}}"""
    elif model_name == "chatglm3-6b":
        input_format = """
        """
    return input_format


@st.cache_resource
def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "chatglm3-6b":
        model = AutoModelForCausalLM.from_pretrained("***/chatglm3-6b", trust_remote_code=True)
        lora_path = "***/LLaMA-Factory/saves/ChatGLM3-6B-Chat/lora/***"
        tokenizer = AutoTokenizer.from_pretrained("***/chatglm3-6b", trust_remote_code=True)
        model.to(device)
    model = PeftModel.from_pretrained(model, lora_path)
    return model, tokenizer


def llm_chat(model_name, model, tokenizer, model_config, query):
    top_p = model_config['top_p']
    max_length = model_config['max_length']
    do_sample = model_config['do_sample']
    temperature = model_config['temperature']

    if model_name == "chatglm3-6b":
        response, _ = model.chat(tokenizer, query, top_p=top_p, max_length=max_length, do_sample=do_sample,
                                       temperature=temperature)
    return response

if __name__ == "__main__":
    user_avator = "🧑‍💻"
    robot_avator = "🤖"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    torch.cuda.empty_cache()
    base_config, model_config = set_config()
    model_name = base_config['model_name']
    model, tokenizer = load_model(model_name=model_name)
    input_format = set_input_format(model_name=model_name)

    st.header(f'Large Language Model：{model_name}')

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

    if user_query := st.chat_input("Please input..."):
        with st.chat_message("user", avatar=user_avator):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query, "avatar": user_avator})
        with st.chat_message("robot", avatar=robot_avator):
            message_placeholder = st.empty()
            query = input_format.replace("{{query}}", user_query)
            max_len = model_config['max_length']
            if len(query) > max_len:
                cur_response = f'Word count exceeds {max_len}, please input again.'
            else:
                cur_response = llm_chat(model_name, model, tokenizer, model_config, query)
            fs.write(f'Input: {query}')
            fs.write('\n')
            fs.write(f'Output: {cur_response}')
            fs.write('\n')
            sys.stdout.flush()
            cur_response = f"""
            {cur_response}
            """
            message_placeholder.markdown(cur_response)
            st.session_state.messages.append({"role": "robot", "content": cur_response, "avatar": robot_avator})