from tensorflow.keras import optimizers, layers,regularizers,models
from tensorflow import keras

from ownmodels import *
        
# HIPERPARÁMETROS - FIJOS
batch_size   = 32
epochs       = 300
patiente     = 50

generator    = True
reduce       = True
windows_size = 1000

lr           = 0.001
opt          = optimizers.Adam(learning_rate = lr)
opt_name     = 'Adam'

# Nombre modelo y del csv a sacar
experimento  =  1000
name_modelo  = 'modelo_1_1000'
csv_name     = 'modelo_'+str(experimento)
notas        = "Reducción de ventana a 5000"
# Fols en Cross-Validation
folds = 3


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

# Cargamos librerías
import numpy as np
from data import DataReader, DataGenerator
from utils import *
from CrossValidation import *
import sys
sys.path.append("S-TSFE-DL")
from TSFEDL.models_keras import *
from tensorflow.keras import optimizers, layers
from tensorflow import keras

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
dict_parameters =  {'input_tensor' :None,
                    'input_shape'  :X.shape[1:],
                    'classes'      :clases,
                    'classifier_activations':'softmax'}



# Creación de modelo
model = modelo_1( input_shape = dict_parameters['input_shape'], 
                  num_classes = dict_parameters['classes'],
                  drop_out = 0.4,
                  filters = 128, 
                  kernel_size = 5,
                  padding = 'same'
                  )






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
                      patiente          = patiente,
                      using_generators  = generator)

# Cross Validation
hist = cv.cross_validate(verbose=1)

scores = cv._evaluate_model(verbose=1)
print("Scores: ", scores)
y_pred, final_pred = cv._predict_model()