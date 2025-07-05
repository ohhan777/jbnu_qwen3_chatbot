import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from typing import List, Tuple, Generator
import threading

class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-8B", enable_thinking=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.history = []
        self.enable_thinking = enable_thinking

    def generate_response(self, user_input) -> Generator[str, None, None]:
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Create streamer for streaming output
        streamer = TextIteratorStreamer(self.tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = {
            **inputs,
            "max_new_tokens": 32768,
            "do_sample": True,
            "temperature": 0.7,
            "streamer": streamer
        }
        
        # Start generation in a separate thread
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream the response
        partial_response = ""
        for new_text in streamer:
            partial_response += new_text
            yield partial_response
        
        # Update history after streaming is complete
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": partial_response})
        
        thread.join()

    def clear_history(self):
        self.history = []

def initialize_chatbot():
    if "chatbot" not in st.session_state:
        with st.spinner("Qwen3-8B 모델을 불러오는 중..."):
            st.session_state.chatbot = QwenChatbot(enable_thinking=False)
        st.success("모델 로딩 완료!")
    return st.session_state.chatbot

def main():
    st.set_page_config(
        page_title="Qwen3-8B Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Qwen3-8B Chatbot")
    st.markdown("Qwen3-8B 모델을 사용한 AI 챗봇입니다.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for controls
    with st.sidebar:
        st.header("설정")
        if st.button("새 채팅", type="secondary"):
            st.session_state.messages = []
            if "chatbot" in st.session_state:
                st.session_state.chatbot.clear_history()
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("메시지를 입력하세요..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Initialize chatbot
            chatbot = initialize_chatbot()
            
            # Stream the response
            full_response = ""
            for partial_response in chatbot.generate_response(prompt):
                full_response = partial_response
                message_placeholder.markdown(full_response + "▌")
            
            # Final response without cursor
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()