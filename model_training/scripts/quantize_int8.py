import os
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

input_model = r"c:\Users\adars\OneDrive\Desktop\PFPD-ll\models\pothole_yolo11n_rdd_v2\weights\best.onnx"
output_model = r"c:\Users\adars\OneDrive\Desktop\PFPD-ll\models\pothole_yolo11n_rdd_v2\weights\best_int8.onnx"

print(f"Quantizing {input_model} to INT8...")
quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False
)

size_orig = os.path.getsize(input_model) / (1024 * 1024)
size_int8 = os.path.getsize(output_model) / (1024 * 1024)

print(f"Original ONNX size: {size_orig:.2f} MB")
print(f"Quantized INT8 ONNX size: {size_int8:.2f} MB")
print(f"Compression ratio: {size_orig / size_int8:.2f}x")

# Verify the quantized model can run an inference session
session = ort.InferenceSession(output_model)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_name = session.get_outputs()[0].name
print(f"Verified Session! Input: {input_name} {input_shape} -> Output: {output_name}")
