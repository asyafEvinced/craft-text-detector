import os
import logging
import numpy as np
from PIL import Image

import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader


class DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder, augmented_model_path='augmented_model.onnx'):
        self.image_folder = calibration_image_folder
        self.augmented_model_path = augmented_model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            session = onnxruntime.InferenceSession(self.augmented_model_path, None)
            (_, _, height, width) = session.get_inputs()[0].shape
            nhwc_data_list = preprocess_func(self.image_folder, height, width, size_limit=0)
            input_name = session.get_inputs()[0].name
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{input_name: nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)


def preprocess_func(images_folder, height, width, size_limit=0):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    image_names = os.listdir(images_folder)
    if 0 < size_limit <= len(image_names):
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))
        input_data = np.float32(pillow_img) - np.array([123.68, 116.78, 103.94], dtype=np.float32)
        tmp_data = np.transpose(input_data, (2, 0, 1))
        nhwc_data = np.expand_dims(tmp_data, axis=0)
        unconcatenated_batch_data.append(nhwc_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data


def run_static_quantization(input_model_path, output_model_path, calibration_dataset_path):
    dr = DataReader(calibration_dataset_path)
    quantize_static(input_model_path, output_model_path, dr, ['Conv', 'QLinearLeakyRelu', 'LeakyRelu'])
    logging.info('Calibrated and quantized model saved.')
