# Read files names
import os


class Cam:
    def __init__(self, image, target, layer, method, file_name):
        self.image = image
        self.target = target
        self.layer = layer
        self.method = method
        self.file_name = file_name


def get_class_files_tuple(dataset_folder):
    '''
    Surf over dataset_folder and return a tuple with the class name and
    a array with each file name inside this class folder

    Return a tuple type like:
    dataset_files([('class_name_1', ['filename1', 'filename2']),
            ('class_name_2', ['filename3', 'filename4'])])

    '''
    dataset_files = {}
    for class_name in os.listdir(dataset_folder):
        if class_name == '.DS_Store':
            continue
        dataset_files[class_name] = []
        for file_name in os.listdir(dataset_folder+class_name+'/'):
            if file_name == '.DS_Store':
                continue
            dataset_files[class_name].append(file_name)
    return dataset_files.items()


def create_folder_if_not_exists(folder):
    '''
    Check if a folder exists, and create it if not exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)


def create_cam_folders(cam_folder, dataset_tuple):
    '''
    This will create a folder for each class inside the `cam_folder`


    cam_folder: ex.: './cam/'
    dataset_tuple: ex.: [('class_name_1', ['filename1', 'filename2']),
            ('class_name_2', ['filename3', 'filename4'])]

    '''
    for class_name, files_names in dataset_tuple:
        create_folder_if_not_exists(os.path.join(cam_folder, class_name))


def save_model_summary(model, file_path):
    '''
    Save model summary

    model: Model to be saved
    file_path: Path to save the model
    '''
    # Print Model on model_summary
    with open(file_path, 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(line_length=100, print_fn=lambda x: fh.write(x + '\n'))


def extract_file_name(file_path):
    base_name = os.path.basename(file_path)
    return os.path.splitext(base_name)[0]
