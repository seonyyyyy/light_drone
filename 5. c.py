# 5.py
# TFLite → C array 변환 (.c 확장자 버전)

input_file = r"C:\Users\jangh\Documents\pbl\model_int8.tflite"
output_file = r"C:\Users\jangh\Documents\pbl\model_data.c"

with open(input_file, "rb") as f:
    data = f.read()

with open(output_file, "w") as f:
    f.write("const unsigned char model_data[] = {")
    for i, b in enumerate(data):
        if i % 12 == 0:
            f.write("\n ")
        f.write(f"{b},")
    f.write("\n};\n")
    f.write(f"const int model_data_len = {len(data)};\n")

print(f"✅ {output_file} 생성 완료, 길이: {len(data)} bytes")
