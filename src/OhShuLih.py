# Cargamos librerías
import numpy as np
from data import DataReader, DataGenerator
from utils import *
from CrossValidation import *
import sys
sys.path.append("S-TSFE-DL")
from TSFEDL.models_keras import *
from tensorflow.keras import optimizers



################################################################################
############################### HIPERPARÁMETROS ################################
################################################################################

# HIPERPARÁMETROS
batch_size   = 128
epochs       = 500
patiente     = 50

generator    = False
reduce       = True
windows_size = 3000

lr           = 0.001
opt          = optimizers.Adam(learning_rate = lr)
opt_name     = 'Adam'

# Nombre modelo y del csv a sacar
experimento  =  1
name_modelo  = 'OhShuLih'
csv_name     = 'OhShuLih_'+str(experimento)

notas        = ""
# Fols en Cross-Validation
folds = 5


dic = {
    'batch_size':batch_size,
    'epochs':epochs,
    'patiente':patiente,
    'n_split':folds,
    
    'generator':generator,
    'reduce':reduce,
    'windows_size':windows_size,
    
    'lr':lr,
    'opt':opt_name,
    
    'name_modelo':name_modelo,
    'csv_name':csv_name,
    'experimento':experimento,
    'notas':notas
    }

################################################################################
################################## MODELO ######################################
################################################################################


# Lectura de datos
data_dir ='dataset/data'
reader = DataReader(data_dir,reduce_dim=reduce,windows_size = windows_size)
X, y = reader.get_samples_labels()

# Imprimimos los parámetros 
clases = len(np.unique(y))
print("Input Shape: \t",X.shape)
dic['classes']     = clases
print_params(dic)

# Se establecen los parámetros del modelo
dict_parameters =  {'include_top'  :True, 
                    'weights'      :None, 
                    'input_tensor' :None,
                    'input_shape'  :X.shape[1:],
                    'classes'      :clases,
                    'classifier_activations':'softmax'}

# Creación de modelo
model = OhShuLih( include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])

# Instanciación del CV
cv = CrossValidation( X                 = X,
                      y                 = y,
                      model             = model,
                      model_name        = name_modelo,
                      csv_name          = csv_name,
                      num_classes       = clases,
                      batch_size        = batch_size,
                      opt               = opt,
                      epochs            = epochs,
                      n_split           = folds,
                      CheckPoint        = False,
                      EarlyStopping     = True,
                      tensorboard       = True,
                      patiente          = patiente,
                      using_generators  = generator)

# Cross Validation
hist = cv.cross_validate(verbose=1)

scores = cv._evaluate_model(verbose=1)
print("Scores: ", scores)
