import vosk
import pyaudio
import json
import os


model_path = r"D:\GR1\speech_to_text\vosk-model-vn-0.4"


if not os.path.exists(model_path):
    raise FileNotFoundError(f"Không tìm thấy thư mục model tại: {model_path}")

model = vosk.Model(model_path)
rec = vosk.KaldiRecognizer(model, 16000) 

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,  
                input=True,
                frames_per_buffer=4096)  

output_file_path = "recognized_text.txt"

with open(output_file_path, "w", encoding="utf-8") as output_file:  
    print("🎙️ Đang lắng nghe... Nói 'dừng' để thoát.")

    while True:
        data = stream.read(4096, exception_on_overflow=False)  
        if len(data) == 0:
            continue

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            recognized_text = result['text']
            
            if recognized_text.strip():
                output_file.write(recognized_text + "\n")
                print("Bạn nói:", recognized_text)
            
            if "dừng" in recognized_text.lower():
                print("⏹️ Dừng chương trình...")
                break

stream.stop_stream()
stream.close()
p.terminate()
