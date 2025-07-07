from typing import List, Tuple, Generator
import threading
import time
import queue

class SimpleTextIteratorStreamer:
    """
    TextIteratorStreamer의 간단한 구현
    yield를 사용하여 스트리밍을 구현
    """
    def __init__(self):
        self.queue = queue.Queue()
        self.finished = False
    
    def put(self, text):
        """생성된 텍스트를 큐에 추가"""
        self.queue.put(text)
    
    def end(self):
        """생성 완료 신호"""
        self.finished = True
    
    def __iter__(self):
        """이터레이터로 동작하여 큐에서 텍스트를 yield"""
        while not self.finished or not self.queue.empty():
            try:
                # 큐에서 텍스트를 가져옴 (0.1초 타임아웃)
                text = self.queue.get(timeout=0.1)
                yield text
            except queue.Empty:
                # 큐가 비어있고 아직 생성 중이면 계속 대기
                if not self.finished:
                    continue
                else:
                    break

def simulate_model_generation(streamer, text_parts):
    """
    모델이 텍스트를 생성하는 것을 시뮬레이션하는 함수
    실제로는 model.generate()가 이 역할을 합니다
    """
    for part in text_parts:
        time.sleep(0.5)  # 각 토큰 생성에 0.5초 소요 시뮬레이션
        streamer.put(part)  # 생성된 토큰을 스트리머에 추가
    
    streamer.end()  # 생성 완료 신호

def main():
    # 생성할 텍스트 부분들 (실제로는 모델이 토큰별로 생성)
    text_parts = ["안녕하세요", "! ", "반갑", "습니다. ","저는 ", "AI ", "챗봇", "입니다", ". ", "어떻게 ", "도와드릴까요", "?"]
    
    # SimpleTextIteratorStreamer 생성
    streamer = SimpleTextIteratorStreamer()
    
    # 모델 생성 스레드 시작
    generation_thread = threading.Thread(
        target=simulate_model_generation,
        args=(streamer, text_parts)
    )
    generation_thread.start()
    
    print("생성 시작... (실시간으로 출력됩니다)")
    print("응답: ", end="", flush=True)
    
    # 스트리머에서 생성된 텍스트를 실시간으로 받아서 출력
    for text in streamer:
        print(text, end="", flush=True)
    
    print("\n생성 완료!")
    generation_thread.join()

if __name__ == "__main__":
    main()