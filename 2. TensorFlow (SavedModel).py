import onnx
from onnx_tf.backend import prepare

# ONNX 모델 파일 경로 지정
onnx_model = onnx.load(r"C:\Users\jangh\Documents\pbl\model.onnx")

# TensorFlow 모델로 변환
tf_rep = prepare(onnx_model)

# SavedModel 형태로 저장
tf_rep.export_graph("C:\\Users\\jangh\\Documents\\pbl\\model_TensorFlow")

print("✅ TensorFlow SavedModel 변환 완료 (saved_model 폴더 생성됨)")
