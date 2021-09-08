# mrOS data locations
dir_annotations = 'D:/mros/annotations-events-nsrr/visit1/'
dir_edfs = 'D:/mros/edfs/visit1/'

# Project directory
project_dir = 'E:/msc_project/data/'

# Training, validation, testing identifiers
train_identifiers_dir = project_dir + 'train_identifiers.npy'
val_identifiers_dir = project_dir + 'val_identifiers.npy'
test_identifiers_dir = project_dir + 'test_identifiers.npy'

# Preprocessed data file directories
train_dir = project_dir + 'train/'
val_dir = project_dir + 'val/'
test_dir = project_dir + 'test/'
val_inference_dir = project_dir + 'val_inference/'

# Confidence directories
val_conf_dir = project_dir + 'conf/val/'
test_conf_dir = project_dir + 'conf/test/'
val_inference_conf_dir = project_dir + 'conf/val_inference/'

# Model save/load directory
model_dir = 'E:/msc_project/model'

# Default network params
params = {'frame_size': 64, 'kernel_size':(5,5), 'filters': 256}
