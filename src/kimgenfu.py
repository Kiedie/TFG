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
batch_size   = 16
epochs       = 300
patiente     = 50

generator    = False
reduce       = True
windows_size = 5000

lr           = 0.001
opt          = optimizers.Adam(learning_rate = lr)
opt_name     = 'Adam'

# Fols en Cross-Validation
folds = 3

# Nombre modelo y del csv a sacar
experimento  =  2
name_modelo  = ['KimTaeYoung','GenMinXing','FujiangMeng']
csv_name     = [[name_modelo[i]+str(experimento)] for i in range(len(name_modelo)) ] 

notas        = "Normalización Ivan"


################################################################################
################################### MODELOS ####################################
################################################################################


# Lectura de datos
data_dir ='dataset/data'
reader = DataReader(data_dir,reduce_dim=reduce,windows_size = windows_size)
X, y = reader.get_samples_labels()
clases = len(np.unique(y))



# Se establecen los parámetros del modelo
dict_parameters =  {'include_top'  :True, 
                    'weights'      :None, 
                    'input_tensor' :None,
                    'input_shape'  :X.shape[1:],
                    'classes'      :clases,
                    'classifier_activations':'softmax'}


modelos = init_kim_gen_fuji(dict_parameters)



for i,clave in enumerate(modelos):
    
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

        'name_modelo':name_modelo[i],
        'csv_name':csv_name[i],
        'experimento':experimento,
        'notas':notas
        }



    # Imprimimos los parámetros 

    print("Input Shape: \t",X.shape)
    dic['classes']     = clases
    print_params(dic)


     
    # Instanciación del CV
    cv = CrossValidation( X                 = X,
                          y                 = y,
                          model             = modelos[clave],
                          model_name        = name_modelo[i],
                          csv_name          = csv_name[i],
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

    del cv