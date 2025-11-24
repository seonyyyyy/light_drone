import tensorflow as tf
import numpy as np

# === 1. 대표 데이터셋 함수 정의 ===
def representative_dataset():
    for _ in range(100):  # 샘플 100개
        # 예시: obs 차원이 39라고 가정 (LiDAR 36 + 거리 1 + 상대위치 2)
        lidar = np.random.uniform(low=0.0, high=60.0, size=(36,))
        dist = np.random.uniform(low=0.0, high=100.0, size=(1,))
        rel_pos = np.random.uniform(low=-50.0, high=50.0, size=(2,))
        obs = np.concatenate([lidar, dist, rel_pos], axis=0).astype(np.float32)
        yield [obs.reshape(1, -1)]

# === 2. SavedModel 로드 및 변환기 생성 ===
converter = tf.lite.TFLiteConverter.from_saved_model(
    r"C:\Users\jangh\Documents\pbl\model_TensorFlow"
)

# === 3. INT8 양자화 옵션 설정 ===
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# === 4. 변환 실행 ===
tflite_model = converter.convert()

# === 5. 저장 ===
with open(r"C:\Users\jangh\Documents\pbl\model_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ INT8 양자화된 model_int8.tflite 생성 완료")
