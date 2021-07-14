import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch_pruning as tp
from yolo.yolo2 import Yolo_Block
from train.train import train_model, RAdam , test_model
from yolo.CSP import ConvBlock
import numpy as np

from loss import Loss

from utils import class_accuracy
from dataset.dataset import get_data
import onnx 
from onnx_tf.backend import prepare
import tensorflow as tf





if __name__ == '__main__':
    



    model = Yolo_Block(3,3,1)
    model.load_state_dict(torch.load('/models/model4.pt'))

    # Conversion PyTorch-ONNX-TF-TFlite
    sample_input = torch.rand((1,3,416, 416)).to("cuda:0")
    torch.onnx.export(
    model,                  
    sample_input,                   
    'onnx_model.onnx',       
    opset_version=12,       
    input_names=['input'] ,  
    output_names=['output1', 'output2'] )

    model = onnx.load("onnx_model.onnx")

    tf_rep = prepare(model)
    tf_rep.export_graph('modeld50_tf')

    converter = tf.lite.TFLiteConverter.from_saved_model('model_tf')
    tflite_model = converter.convert()
    with open('prunded.tflite', 'wb') as f:
        f.write(tflite_model)

    