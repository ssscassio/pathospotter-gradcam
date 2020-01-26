from util import get_class_files_tuple
from model import load_model
from image import load_image, preprocess_image
import numpy as np

def main():
    # 1. Carrega o modelo
    model_path = './models/best_modelFold5' 
    model = load_model(model_path)
    
    # 2. Pasta contendo os arquivos de imagem
    dataset_folder = './dataset/'
    dataset_tuple = get_class_files_tuple(dataset_folder)
      
    # Iterando sobre o dicionário
    for class_name, files_names in dataset_tuple:
        print('-------Processing class: ',class_name,' -------')
        for image_name in files_names:
            image_path = dataset_folder+class_name+'/'+image_name
     
            classes = {
                    'com': 0,
                    'sem': 1,
                    }
            # Load image
            image = load_image(image_path)
            height, width, _ = image.shape
            
            # Preprocess Image
            _, input_width, input_height, _ = model.layers[0].input_shape # Get model's input shape
            preprocessed_image = preprocess_image(image, (input_width, input_height))
            
            
            # Predict Image class
            predictions = model.predict(preprocessed_image)
            predicted_class = np.argmax(predictions)

            if predictions[0][predicted_class] <= 0.8 or predicted_class != classes[class_name] :
                print('Image name: ' + image_path)
                print("Prediction: with(" + str(predictions[0][0]) + ')  without('+ str(predictions[0][1])+')')
                print(' *** Done ***\n')

if __name__ == '__main__':
    main()