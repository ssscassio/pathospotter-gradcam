from model import load_model, get_model_layers, get_model_nb_classes
from util import save_model_summary, Cam
from image import load_image, preprocess_image, save_image
from gradcam import grad_cam, counterfactual_explanation
#from guided_gradcam import guided_grad_cam
import numpy as np
from deprocess import Method, create_cam_image


# class Method(Enum):
#    CAM_IMAGE_JET = 0
#    CAM_IMAGE_BONE = 1
#    CAM_AS_WEIGHTS = 2
#    JUST_CAM_JET = 3
#    JUST_CAM_BONE = 4


# Loads the model
model_path = './models/glomeruloesclerose'
model = load_model(model_path)

# Save model summary into file
model_summary_path = model_path + '_summary.txt'
save_model_summary(model, model_summary_path)

# Load image
image_folder = './examples/'
image_name = 'without.png'
image_path = image_folder + image_name
image = load_image(image_path)
height, width, _ = image.shape

# Preprocess Image
# Get model's input shape
_, input_width, input_height, _ = model.layers[0].input_shape
preprocessed_image = preprocess_image(image, (input_width, input_height))

# Layers to be visualized
layers = get_model_layers(model)

# layers = layers[0:-8] # Exclude last 8 layers because they are dense, or kernel size (1,1)
# Take only convolutional deep layers (That seems to be relevant)
layers = layers[9:-8]

# Model's number of classes
nb_classes = get_model_nb_classes(model)

# Image prediction probabilities and predicted_class
# [1,0] = with
# [0,1] = without
predictions = model.predict(preprocessed_image)
predicted_class = np.argmax(predictions)

# Which class use to visualization
class_name = ['with', 'without']
class_to_visualize = predicted_class  # You can change it to a specific class

# Print prediction info
print("Class_to_visualize rate: " +
      str(predictions[0][class_to_visualize]) + ' to Class: (' + class_name[class_to_visualize] + ')')


method_name = ['CAM_IMAGE_JET', 'CAM_IMAGE_BONE',
               'CAM_AS_WEIGHTS', 'JUST_CAM_JET', 'JUST_CAM_BONE']

process = 1
if process == 0:  # Visualize specific layer with specific visualization method
    # Select Layer to visualize
    # layer_to_visualize = 'conv2d_41' | 'max_pooling2d_33'
    layer_to_visualize = 'max_pooling2d_33'

    # Get cam
    cam = grad_cam(model, preprocessed_image, class_to_visualize,
                   layer_to_visualize, nb_classes)

    # Apply visualization method
    method = Method.CAM_IMAGE_JET
    cam_heatmap = create_cam_image(cam, image, method)

    save_image(cam_heatmap, './experiments/'+image_name)
elif process == 1:  # Merge change target experiment
    #cams = []
    for layer_to_visualize in layers:
        # Get cam
        cam = grad_cam(model, preprocessed_image,
                       class_to_visualize, layer_to_visualize, nb_classes)

        # Apply visualization method
        method = Method.CAM_IMAGE_JET
        cam_heatmap = create_cam_image(cam, image, method)

        #cams.append(Cam(cam_heatmap, class_name[class_to_visualize], layer_to_visualize, method_name[method], image_name))
        save_image(cam_heatmap, './output/'+layer_to_visualize+'.png')
