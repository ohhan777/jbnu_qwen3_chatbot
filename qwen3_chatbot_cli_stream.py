from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-8B", enable_thinking=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.history = []
        self.enable_thinking = enable_thinking

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        output = self.model.generate(**inputs, max_new_tokens=32768, streamer=streamer)
        response_ids = output[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

def main():
    print("Qwen3-8B 모델을 불러오는 중입니다... (최초 실행 시 시간이 걸릴 수 있습니다)")
    chatbot = QwenChatbot(enable_thinking=False)
    print("Qwen3-8B 챗봇에 오신 것을 환영합니다! (종료하려면 'exit' 입력)")
    
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("챗봇을 종료합니다.")
            break
        
        print("Qwen3: ", end="", flush=True)
        chatbot.generate_response(user_input)
        print()  # Add newline after streaming

if __name__ == "__main__":
    main()
