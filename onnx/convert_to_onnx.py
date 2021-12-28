import numpy as np
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


def test_output(torch_out, x):
    ort_session = onnxruntime.InferenceSession(ONNX_MODEL_NAME)

    def to_numpy(tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        else:
            tensor.cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    for i,j in zip(torch_out, ort_outs):
        np.testing.assert_allclose(to_numpy(i), j, rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


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
    test_output(torch_out, x)


if __name__ == '__main__':
    main()
