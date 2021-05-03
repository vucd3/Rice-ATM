import tensorflow as tf
from tensorflow.keras.applications import MobileNet
import numpy as np
class MobileNetClass():
    def __init__(self,input_shape=(32,32,3),alpha=0.25):
        mobilenet_model = MobileNet(input_shape=input_shape,alpha=alpha,include_top=False)
        self.layer = mobilenet_model.get_layer("conv_pw_13_relu")
        truncatedModel = tf.keras.Model(inputs=mobilenet_model.inputs,outputs=self.layer.output)
        self.model = tf.keras.Sequential()
        self.model.add(truncatedModel)
        self.model.add(tf.keras.layers.Flatten())
    def GetModel(self):
        return self.model
    def GetInputShape(self):
        inputShape = self.layer.output.shape.as_list()[1:]
        return tf.TensorShape(inputShape)
mb = MobileNetClass()
model = mb.GetInputShape()
#print(model)