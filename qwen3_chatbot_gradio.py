import gradio as gr
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

# Global chatbot instance
chatbot = None

def initialize_chatbot():
    global chatbot
    if chatbot is None:
        print("Qwen3-8B 모델을 불러오는 중...")
        chatbot = QwenChatbot(enable_thinking=False)
        print("모델 로딩 완료!")
    return chatbot

def chat_function(message: str, history: List[Tuple[str, str]]) -> Generator[Tuple[str, List[Tuple[str, str]]], None, None]:
    bot = initialize_chatbot()
    
    # Add user message to history immediately
    history.append((message, ""))
    
    # Stream the response
    for partial_response in bot.generate_response(message):
        # Update the last message in history with the streaming response
        history[-1] = (message, partial_response)
        yield "", history

def start_new_chat():
    global chatbot
    if chatbot:
        chatbot.clear_history()
    return [], ""

def create_interface():
    with gr.Blocks(title="Qwen3-8B Chatbot") as demo:
        gr.Markdown("# 🤖 Qwen3-8B Chatbot")
        gr.Markdown("Qwen3-8B 모델을 사용한 AI 챗봇입니다.")
        
        with gr.Row():
            with gr.Column(scale=1):
                new_chat_button = gr.Button("새 채팅", variant="secondary", size="sm")

        with gr.Row():
            with gr.Column(scale=4):
                chatbot_interface = gr.Chatbot(
                    label="대화",
                    height=500,
                    show_copy_button=True
                )
            
        with gr.Row():
            with gr.Column(scale=4):
                msg = gr.Textbox(
                    label="메시지",
                    placeholder="메시지를 입력하고 엔터를 누르세요...",
                    lines=1
                )
        
        # Event handlers - only enter key submission with streaming
        msg.submit(
            chat_function,
            inputs=[msg, chatbot_interface],
            outputs=[msg, chatbot_interface],
            show_progress=True
        )

        new_chat_button.click(
            start_new_chat,
            inputs=None,
            outputs=[chatbot_interface, msg], # 채팅창과 메시지 입력창을 초기화
            queue=False
        )
    
    return demo

def main():
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    main()