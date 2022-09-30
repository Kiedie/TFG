# Cargamos librerías
from tensorflow.keras import optimizers, layers,regularizers,models
from tensorflow import keras
from ownmodels import *
import numpy as np
from data import DataReader, DataGenerator
from utils import *
from CrossValidation import *
import sys
sys.path.append("S-TSFE-DL")
from TSFEDL.models_keras import *
from tensorflow.keras import optimizers, layers
from tensorflow import keras

# HIPERPARÁMETROS
batch_size   = 64
epochs       = 300
patiente     = 50

generator    = False
reduce       = True
windows_size = 3000

lr           = 0.001
opt          = optimizers.Adam(learning_rate = lr)
opt_name     = 'Adam'

# Nombre modelo y del csv a sacar
experimento  =  1
name_modelo  = 'modelo_1_experimentacion_parte_3'
csv_name     = 'modelo_1_experimentacion_parte_3.csv'

notas        = "Ejecutar este modelo sin reducir la dimension, ya que en caso contrario, falla por el pooling"
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

tuning = { 'drop_out':[0.4],
           'filters': [256],
           'kernel_size':[9],
           'padding':['same']}






name = 'modelo_1_experimentacion_parte_3.csv'


results = { 'drop':              [],
            'filters':           [],
            'kernel':            [],
            'padding':           [],
            'Test loss':         [],
            'Test accuracy':     [],
            'Test sco':          [],
            'Normal ECGs score': [],
            'AF ECGs score':     [],
            'Other ECGS score':  [],
            'Noisy ECGS score':  [],
            'Total Score':       []}

for drop in tuning['drop_out']:
    for filt in tuning['filters']:
        for kernel in tuning['kernel_size']:
            for padding in tuning['padding']:

                results['drop'].append(drop)
                results['filters'].append(filt)
                results['kernel'].append(kernel)
                results['padding'].append(padding)
                

                model = modelo_1( input_shape = dict_parameters['input_shape'], 
                  num_classes = dict_parameters['classes'],
                  drop_out = drop,
                  filters = filt, 
                  kernel_size = kernel,
                  padding = padding
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
                                    using_generators  = generator,
                                    tensorboard       = True)




                # Cross Validation
                hist,scores = cv.cross_validate(ret_results = True,verbose=1)
                
              
                results['Test accuracy'].append(scores['Test accuracy'])
                results['Test sco'].append(scores['Test sco'])
                results['Test loss'].append(scores['Test loss'])
                results['Normal ECGs score'].append(scores['Normal ECGs score'])
                results['AF ECGs score'].append(scores['AF ECGs score'])
                results['Other ECGS score'].append(scores['Other ECGS score'])
                results['Noisy ECGS score'].append(scores['Noisy ECGS score'])
                results['Total Score'].append(scores['Total Score'])
                #scores =  cv.get_results(final_pred)
                #scores = cv._evaluate_model(verbose=1)
                
                print("Scores: ", results)

                pd.DataFrame(results).to_csv(name)

                
