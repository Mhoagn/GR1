import joblib

pipe_lr = joblib.load("text_emotion.pkl")
text_test = ["I am afraid of dark"]
predicted_label = pipe_lr.predict(text_test)
print("Kết quả dự đoán:", predicted_label[0])


