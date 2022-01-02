import argparse
import onnx


def parse_args():
    parser = argparse.ArgumentParser(description='Run onnx model on data')
    parser.add_argument("-m", "--model", required=True, help="Model path")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    onnx_model = onnx.load(args.model)


if __name__ == '__main__':
    main()
