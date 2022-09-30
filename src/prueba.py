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
batch_size   = 32
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
name_modelo  = 'prueba'
csv_name     = 'prueba'+str(experimento)

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

tuning = { 'drop_out':[0.2,0.4,0.6],
           'filters': [128,64,32],
           'kernel_size':[3,5],
           'padding':['same','valid']}






name = 'prueba.csv'


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
                

                model = modelo_2( input_shape = dict_parameters['input_shape'], 
                  num_classes = dict_parameters['classes']
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
                                    tensorboard       = False)




                # Cross Validation
                hist,scores = cv.cross_validate(ret_results = True,verbose=1)
                
                results['Test loss']         = scores['Test loss']
                results['Test accuracy']     = scores['Test accuracy']
                results['Test sco']          = scores['Test sco']
                results['Test loss']         = scores['Test loss']
                results['Normal ECGs score'] = scores['Normal ECGs score']
                results['AF ECGs score']     = scores['AF ECGs score']
                results['Other ECGS score']  = scores['Other ECGS score']
                results['Noisy ECGS score']  = scores['Noisy ECGS score']
                results['Total Score']       = scores['Total Score']
                #scores =  cv.get_results(final_pred)
                #scores = cv._evaluate_model(verbose=1)
                
                print("Scores: ", results)

                pd.DataFrame(results).to_csv(name)

                