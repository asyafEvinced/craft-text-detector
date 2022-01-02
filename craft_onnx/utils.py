import numpy as np
import onnxruntime


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        tensor.cpu().numpy()


def run_onnx_model(onnx_model, input_data, input_is_fp16=False):
    ort_session = onnxruntime.InferenceSession(str(onnx_model))

    numpy_data = to_numpy(input_data)
    if input_is_fp16:
        numpy_data = numpy_data.astype(np.float16)
    ort_inputs = {ort_session.get_inputs()[0].name: numpy_data}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs
