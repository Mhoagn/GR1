import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import queue
import datetime
import os

class AudioRecorder:
    def __init__(self):
        self.model = whisper.load_model("base")
        self.recording = False
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        
        # Tạo thư mục để lưu file âm thanh và văn bản
        if not os.path.exists("recordings"):
            os.makedirs("recordings")
        if not os.path.exists("transcripts"):
            os.makedirs("transcripts")

    def callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_queue.put(indata.copy())

    def start_recording(self):
        self.recording = True
        self.audio_data = []
        
        # Bắt đầu ghi âm trong một luồng riêng
        self.stream = sd.InputStream(
            callback=self.callback,
            channels=1,
            samplerate=self.sample_rate
        )
        self.stream.start()
        print("Đang ghi âm... Nhấn Enter để dừng.")

    def stop_recording(self):
        self.recording = False
        self.stream.stop()
        self.stream.close()
        
        # Xử lý dữ liệu âm thanh đã ghi
        while not self.audio_queue.empty():
            self.audio_data.append(self.audio_queue.get())
        
        if len(self.audio_data) > 0:
            audio_data = np.concatenate(self.audio_data)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Lưu file âm thanh
            audio_filename = f"recordings/recording_{timestamp}.wav"
            wav.write(audio_filename, self.sample_rate, audio_data)
            
            # Nhận dạng giọng nói
            print("Đang xử lý âm thanh...")
            result = self.model.transcribe(audio_filename, language="vi")
            
            # Lưu văn bản
            text_filename = f"transcripts/transcript_{timestamp}.txt"
            with open(text_filename, "w", encoding="utf-8") as f:
                f.write(result["text"])
            
            print("\nKết quả nhận dạng:")
            print(result["text"])
            print(f"\nĐã lưu âm thanh tại: {audio_filename}")
            print(f"Đã lưu văn bản tại: {text_filename}")

def main():
    recorder = AudioRecorder()
    print("Chương trình nhận dạng giọng nói sử dụng Whisper")
    print("-----------------------------------------------")
    
    while True:
        try:
            input("Nhấn Enter để bắt đầu ghi âm...")
            recorder.start_recording()
            input()  # Đợi người dùng nhấn Enter để dừng
            recorder.stop_recording()
            
            choice = input("\nBạn có muốn tiếp tục ghi âm không? (y/n): ")
            if choice.lower() != 'y':
                break
                
        except KeyboardInterrupt:
            print("\nĐã dừng chương trình.")
            break

if __name__ == "__main__":
    main() 