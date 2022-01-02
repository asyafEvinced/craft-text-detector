import numpy as np
import logging
import onnx
from onnxconverter_common import convert_float_to_float16_model_path
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
from PIL import Image
import sys
import torch
from torchvision import transforms

from craft_text_detector import Craft
from craft_onnx.utils import run_onnx_model, to_numpy

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

ONNX_EXT = '.onnx'
ONNX_MODEL_NAME = 'craft'
ONNX_MODEL_FILE = f'{ONNX_MODEL_NAME}{ONNX_EXT}'
IMG_PATH = '../figures/idcard.png'


def test_onnx_model(model_path, input_data, is_fp16=False):
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    logging.info(f'Model check of {model_path} passed')
    run_onnx_model(model_path, input_data, is_fp16)
    logging.info(f'Model run successfully')


def convert_to_onnx(model, input_data, output_model_path):
    torch.onnx.export(model,
                      input_data,
                      str(output_model_path),
                      export_params=True,
                      opset_version=13,  # conversion to flp16 doesn't work with lower opset
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})
    test_onnx_model(output_model_path, input_data)


def test_output(torch_out, input_data, onnx_model):
    ort_outs = run_onnx_model(onnx_model, input_data)
    for i, j in zip(torch_out, ort_outs):
        np.testing.assert_allclose(to_numpy(i), j, rtol=1e-03, atol=1e-05)

    logging.info("Exported model has been tested with ONNXRuntime, and the result looks good!")


def quantize_model_dynamic(onnx_model_path, quantization, input_data):
    model_name = onnx_model_path.stem
    quantized_model_file = f'{model_name}_quant_{str(quantization)}{ONNX_EXT}'
    quantize_dynamic(onnx_model_path,
                     Path(quantized_model_file),
                     weight_type=quantization)
    test_onnx_model(quantized_model_file, input_data)
    logging.info(f"Dynamically quantized model saved to: {quantized_model_file}")


def quantize_model_fp16(onnx_model_path, input_data):
    new_onnx_model = convert_float_to_float16_model_path(onnx_model_path)
    model_name = onnx_model_path.stem
    quantized_model_file = f'{model_name}_quant_fp16{ONNX_EXT}'
    onnx.save(new_onnx_model, quantized_model_file)
    test_onnx_model(quantized_model_file, input_data, True)
    logging.info(f"Fp16 quantized model saved to: {quantized_model_file}")


def run_quantizations(onnx_model_path, input_data):
    quantize_model_dynamic(onnx_model_path, QuantType.QUInt8, input_data)
    # TODO: find why this conversion fails
    # quantize_model_dynamic(onnx_model_path, QuantType.QInt8, input_data)
    quantize_model_fp16(onnx_model_path, input_data)


def load_craft_model():
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
    return model


def load_image_to_tensor(image_path):
    img = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tensor = transform(img)
    tensor = tensor[None, :]
    tensor.requires_grad = True
    return tensor


def main():
    model = load_craft_model()
    input_data = load_image_to_tensor(IMG_PATH)
    torch_out = model(input_data)
    onnx_model_path = Path(ONNX_MODEL_FILE)
    convert_to_onnx(model, input_data, onnx_model_path)
    test_output(torch_out, input_data, onnx_model_path)
    run_quantizations(onnx_model_path, input_data)


if __name__ == '__main__':
    main()
