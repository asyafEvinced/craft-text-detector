import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
import torch

from craft_text_detector import Craft

ONNX_EXT = '.onnx'
ONNX_MODEL_NAME = 'craft'
ONNX_MODEL_FILE = f'{ONNX_MODEL_NAME}{ONNX_EXT}'


def convert(model, input_data, output_model_path):
    torch.onnx.export(model,
                      input_data,
                      str(output_model_path),
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    onnx_model = onnx.load(ONNX_MODEL_FILE)
    onnx.checker.check_model(onnx_model)


def test_output(torch_out, input_data, onnx_model):
    ort_session = onnxruntime.InferenceSession(str(onnx_model))

    def to_numpy(tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        else:
            tensor.cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_data)}
    ort_outs = ort_session.run(None, ort_inputs)

    for i, j in zip(torch_out, ort_outs):
        np.testing.assert_allclose(to_numpy(i), j, rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def quantize_model_dynamic(onnx_model_path, quantization):
    model_name = onnx_model_path.name
    quantized_model_file = f'{model_name}_quant_{str(quantization)}{ONNX_EXT}'
    quantize_dynamic(onnx_model_path,
                     Path(quantized_model_file),
                     weight_type=quantization)

    print(f"Dynamically quantized model saved to: {quantized_model_file}")


def run_quantizations(onnx_model_path):
    quantize_model_dynamic(onnx_model_path, QuantType.QInt8)
    quantize_model_dynamic(onnx_model_path, QuantType.QUInt8)


def main():
    craft = Craft(
        output_dir=None,
        rectify=True,
        export_extra=False,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=False,
        long_size=720,
        refiner=False,
        crop_type="poly",
    )

    model = craft.craft_net

    model.eval()

    batch_size = 1
    input_data = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    torch_out = model(input_data)
    onnx_model_path = Path(ONNX_MODEL_FILE)
    convert(model, input_data, onnx_model_path)
    test_output(torch_out, input_data, onnx_model_path)
    run_quantizations(onnx_model_path)


if __name__ == '__main__':
    main()
