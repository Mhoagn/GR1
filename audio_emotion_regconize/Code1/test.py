import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import librosa

# Định nghĩa các emotion
emotions = {
    'happy': 0,
    'sad': 1, 
    'angry': 2,
    'fear': 3,
    'disgusted': 4,
    'surprised': 5,
    'neutral': 6
}

def extract_features(file_path):
    try:
        # Load file âm thanh
        audio, sr = librosa.load(file_path, duration=3, offset=0.5)
        
        # Trích xuất MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        return mfcc_scaled
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

def test_model():
    # Load model
    print("Đang load model...")
    model = load_model('emotion_recognition_model11.h5')
    print("Model đã được load thành công.")
    
    # Đường dẫn đến thư mục test
    test_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataTest')
    
    # Danh sách để lưu kết quả
    results = []
    
    print("Bắt đầu dự đoán...")
    # Duyệt qua các file trong thư mục test
    for dirpath, dirnames, filenames in os.walk(test_directory):
        for filename in filenames:
            if filename.endswith('.wav'):
                file_path = os.path.join(dirpath, filename)
                file_path = os.path.normpath(file_path).replace('\\', '/')
                
                # Lấy emotion thực tế từ đường dẫn
                true_emotion = None
                for emotion in emotions.keys():
                    if emotion in dirpath.lower():
                        true_emotion = emotion
                        break
                
                # Trích xuất đặc trưng
                features = extract_features(file_path)
                
                if features is not None:
                    # Reshape features
                    features = features.reshape(1, features.shape[0], 1)
                    
                    # Dự đoán
                    prediction = model.predict(features)
                    predicted_idx = np.argmax(prediction)
                    
                    # Chuyển index thành tên emotion
                    predicted_emotion = list(emotions.keys())[list(emotions.values()).index(predicted_idx)]
                    
                    # Lưu kết quả
                    results.append({
                        'file': file_path,
                        'true_emotion': true_emotion,
                        'predicted_emotion': predicted_emotion,
                        'correct': true_emotion == predicted_emotion
                    })
    
    # Tạo DataFrame từ kết quả
    results_df = pd.DataFrame(results)
    
    # Tính độ chính xác
    accuracy = (results_df['correct'].sum() / len(results_df)) * 100
    
    # In kết quả
    print("\nKết quả dự đoán:")
    print(f"Tổng số file test: {len(results_df)}")
    print(f"Độ chính xác: {accuracy:.2f}%")
    
    # Lưu kết quả
    results_df.to_csv('test_results.csv', index=False)
    print("\nĐã lưu kết quả chi tiết vào file test_results.csv")

test_model()

