import onnx
import onnxruntime
import torch

from craft_text_detector import Craft

ONNX_MODEL_NAME = 'craft.onnx'


def convert(model, x):
    torch.onnx.export(model,
                      x,
                      ONNX_MODEL_NAME,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    onnx_model = onnx.load(ONNX_MODEL_NAME)
    onnx.checker.check_model(onnx_model)


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
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    torch_out = model(x)
    convert(model, x)


if __name__ == '__main__':
    main()
