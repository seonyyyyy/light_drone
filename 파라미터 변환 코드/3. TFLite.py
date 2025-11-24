import tensorflow as tf

# SavedModel 불러오기
converter = tf.lite.TFLiteConverter.from_saved_model(r"C:\Users\jangh\Documents\pbl\model_TensorFlow")

# (선택) 최적화 적용
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 변환 실행
tflite_model = converter.convert()

# 결과 저장
with open(r"C:\Users\jangh\Documents\pbl\model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite 변환 완료 (model.tflite 저장됨)")
