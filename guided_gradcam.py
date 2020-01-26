# -*- coding: utf-8 -*-

from tensorflow.python.framework import ops
import tensorflow as tf
import keras.backend as K
import cv2
import numpy as np

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def guided_grad_cam(model, cam, layer_name, image_to_evaluate, width, height):
    cam_heatmap = cv2.resize(cam, (224, 224)) # Resize to input shape using bi-linear interpolation
    register_gradient()
    saliency_fn = compile_saliency_function(model, layer_name)
    saliency = saliency_fn([image_to_evaluate, 0])
    gradcam = saliency[0] * cam_heatmap[..., np.newaxis]
    
    '''Begin of Normalization steps'''
    if np.ndim(gradcam) > 3:
        gradcam = np.squeeze(gradcam)
    # normalize tensor: center on 0., ensure std is 0.1
    gradcam -= gradcam.mean()
    gradcam /= (gradcam.std() + 1e-5)
    gradcam *= 0.1
    
    # clip to [0, 1]
    gradcam += 0.5
    gradcam = np.clip(gradcam, 0, 1)
    
    # convert to RGB array
    gradcam *= 255
    if K.image_dim_ordering() == 'th':
        gradcam = gradcam.transpose((1, 2, 0))
    gradcam = np.clip(gradcam, 0, 255).astype('uint8')
    '''End of Normalization steps'''

    return gradcam