import speech_recognition as sr
import pyttsx3

r = sr.Recognizer()

def record_text():
    try:
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration=0.2)
            print("Đang lắng nghe...")
            audio2 = r.listen(source2)
            myText = r.recognize_google(audio2, language="vi-VN")  
            print(f"Bạn nói: {myText}")
            return myText
    except sr.RequestError as e:
        print(f"Lỗi kết nối: {e}")
    except sr.UnknownValueError:
        print("Không nhận diện được, vui lòng thử lại!")
    return None  

def output_text(text):
    if text:
        with open("output.txt", "a", encoding="utf-8") as f:  
            f.write(text + "\n")
        print(f"Đã lưu: {text}")

while True:
    text = record_text()
    if text:
        if "dừng" in text.lower():
            print("Đang thoát chương trình...")
            break
        output_text(text)
