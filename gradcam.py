import keras.backend as K
from keras.models import Model
from keras.layers.core import Lambda
import tensorflow as tf
import numpy as np


def target_category_loss(x, category_index, nb_classes):
    '''
        Multiply tensor x  by Tensor K.one_hot

    '''
    # Multiply tensor x  by Tensor K.one_hot
    # K.one_hot peforms a computes a one_hot representation for a integer tensor
    # https://www.tensorflow.org/api_docs/python/tf/keras/backend/one_hot
    # For example, for a output like 0 to without lesion and 1 to with lesion
    # One hot peforms a map to [1,0] and [0,1] as output categories of the network
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def _compute_gradients(tensor, var_list):
    grads = K.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def grad_cam(model, image, target_class, layer_name, nb_classes):
    # Create a lambda function, can take any number of arguments but
    # can only have one expression
    def target_layer(x): return target_category_loss(
        x, target_class, nb_classes)

    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(
        model.output)

    new_model = Model(inputs=model.input, outputs=x)

    loss = K.sum(new_model.output)

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in new_model.layers])

    conv_output = layer_dict[layer_name].output

    grads = normalize(K.gradients(loss, conv_output)[0])

    gradient_function = K.function([new_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))  # alpha_k
    # New array with the output shape, but with zeros
    cam = np.zeros(output.shape[0: 2], dtype=np.float32)

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = np.maximum(cam, 0)  # Passing through ReLU
    # Normalization scale to [np.min(cam)/np.max(cam),1]
    # There is a fallback if np.max equal to 0
    cam = cam if np.max(cam) == 0 else cam / np.max(cam)

    return cam


def counterfactual_explanation(model, image, target_class, layer_name, nb_classes):
    # Create a lambda function, can take any number of arguments but
    # can only have one expression
    def target_layer(x): return target_category_loss(
        x, target_class, nb_classes)

    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(
        model.output)

    new_model = Model(inputs=model.input, outputs=x)

    loss = K.sum(new_model.output)

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in new_model.layers])

    conv_output = layer_dict[layer_name].output

    grads = normalize(K.gradients(loss, conv_output)[0])

    gradient_function = K.function([new_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    # The grads_val * -1 is the slight difference between grad-cam and counterfactual_explanations
    weights = np.mean(grads_val * -1, axis=(0, 1))  # alpha_k
    # New array with the output shape, but with zeros
    cam = np.zeros(output.shape[0: 2], dtype=np.float32)

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = np.maximum(cam, 0)  # Passing through ReLU
    # Normalization scale to [np.min(cam)/np.max(cam),1]
    # There is a fallback if np.max equal to 0
    cam = cam if np.max(cam) == 0 else cam / np.max(cam)

    return cam
