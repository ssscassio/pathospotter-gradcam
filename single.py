# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from operator import itemgetter
import sys
import argparse
import os
from guided_gradcam import guided_grad_cam
from model import load_model, get_model_viewable_layers, get_model_nb_classes
from util import save_model_summary, Cam, create_folder_if_not_exists, extract_file_name
from image import load_image, preprocess_image, save_image
from gradcam import grad_cam, counterfactual_explanation
import numpy as np
from deprocess import Method, create_cam_image, create_guided_cam_image, convert_to_bgr, convert_to_rgb, plot


def main(image_file, model_file, layer_name, label, method_name, output_path, guided, no_plot):
    print(image_file, model_file, layer_name, label)

    # 1. Load model
    model = load_model(model_file)
    # 1.1 Save model summary into file
    model_summary_file = model_file + '_summary.txt'
    save_model_summary(model, model_summary_file)

    # 2. Load image
    image = load_image(image_file)
    height, width, _ = image.shape
    # Get model's input shape
    _, input_width, input_height, _ = model.layers[0].input_shape
    # 2.1 Preprocess Image
    preprocessed_image = preprocess_image(image, (input_width, input_height))

    # 3. Layers to visualize
    all_layers = get_model_viewable_layers(model)
    if layer_name == 'all':
        layers = all_layers
    elif layer_name in all_layers:
        layers = [layer_name]
    else:
        print('Error: Invalid layer name')
        return

    # 4. Image prediction probabilities and predicted_class
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions)

    # 5 Label to visualize
    nb_classes = get_model_nb_classes(model)  # 5.1 Model's number of classes
    if label == -1:
        class_to_visualize = predicted_class
    elif label < nb_classes and label > -1:
        class_to_visualize = label
    else:
        print('Error: Invalid label value')
        return

    # 6. Choose Visualization method
    all_methods = ['CAM_IMAGE_JET', 'CAM_IMAGE_BONE',
                   'CAM_AS_WEIGHTS', 'JUST_CAM_JET', 'JUST_CAM_BONE']
    if method_name in all_methods:
        method = Method[method_name]
    elif guided:
        method_name = 'GUIDED'
    else:
        print('Error: Invalid visualization method')
        return

    # 7. Create output folder
    create_folder_if_not_exists(output_path)

    # TODO: Handler with dataset folder
    cams = []
    # 8. Iterate over layers to visualize
    for layer_to_visualize in layers:
        # 8.1 Get cam
        cam = grad_cam(model, preprocessed_image,
                       class_to_visualize, layer_to_visualize, nb_classes)

        # 8.2 Generate visualization
        if guided:
            cam = guided_grad_cam(
                model, cam, layer_to_visualize, preprocessed_image)
            cam_heatmap = create_guided_cam_image(cam, image, cam_rate=1)
        else:
            cam_heatmap = create_cam_image(cam, image, method)

        cams.append(Cam(image=cam_heatmap, target=class_to_visualize,
                        layer=layer_to_visualize, method=method_name, file_name=image_file))

    cams.insert(0, Cam(image=convert_to_bgr(image), target=class_to_visualize,
                       layer='Original', method=method_name, file_name=image_file))
    if no_plot:
        file_name = extract_file_name(image_file)
        create_folder_if_not_exists(os.path.join(output_path, file_name))
        for _cam in cams:
            save_image(_cam.image,
                       os.path.join(output_path, file_name)+'/'+_cam.layer+'.png')
    else:
        plot(cams, image_file)
        plt.savefig(output_path+'/'+extract_file_name(image_file)+'.png')
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This is the PathoSpotter Feature visualize program")
    # Define argument list
    parser.add_argument('--image_file', type=str,
                        help="Path to the input image. Default is './examples/without.png'")
    parser.add_argument('--model_file', type=str,
                        help="Path to the model file for the CNN model. Default is './models/glomeruloesclerose'")
    parser.add_argument('--layer_name', type=str,
                        help="Layer to use for grad-CAM. Default is 'all'.")
    parser.add_argument('--label', type=int,
                        help="Class label to generate grad-CAM for, -1 = use predicted class")
    parser.add_argument('--method', type=str,
                        help="Method used to visualize the grad-CAM. Default is 'CAM_IMAGE_JET'")
    parser.add_argument('--output_path', type=str,
                        help="Path to save images in. Default is './output'")
    parser.add_argument("--guided", action='store_true',
                        help="Activate guided method.")
    parser.add_argument("--no_plot", action='store_true',
                        help="Disactivate plot output. Will generate one file for each layer to visualize")

    # Set arguments Defaults
    parser.set_defaults(
        image_file="./examples/without.png",
        model_file="./models/glomeruloesclerose",
        layer_name="all",
        label=-1,
        method='CAM_IMAGE_JET',
        output_path='./output'
    )
    args = parser.parse_args(sys.argv[1:])
    args = vars(args)

    main(
        image_file=args['image_file'],
        model_file=args['model_file'],
        layer_name=args['layer_name'],
        label=args['label'],
        method_name=args['method'],
        output_path=args['output_path'],
        guided=args['guided'],
        no_plot=args['no_plot']
    )
