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
batch_size   = 128
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
name_modelo  = 'modelo2_experimentacion_parte_2'
csv_name     = 'modelo2_experimentacion__parte_2'

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
           'lstm_units': [16,8,4],
           'dense_units':[32,8],
           'padding':['same']}

name = 'modelo_2_experimentacion_parte_2.csv'


results = {
            'lstm_units':[],
            'dense_units':[],
            'padding':[],
            'Test loss':         [],
            'Test accuracy':     [],
            'Test sco':          [],
            'Normal ECGs score': [],
            'AF ECGs score':     [],
            'Other ECGS score':  [],
            'Noisy ECGS score':  [],
            'Total Score':       []}
count = 0

for lstm in tuning['lstm_units']:
    for dense in tuning['dense_units']:
        for padding in tuning['padding']:
                
            count +=1


            results['lstm_units'].append(lstm)
            results['dense_units'].append(dense)
            results['padding'].append(padding)
                

            model = modelo_2( input_shape = dict_parameters['input_shape'], 
            num_classes = dict_parameters['classes'],
            lstm_units = lstm, 
            dense_units = dense,
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
                            tensorboard       = False)


            # Cross Validation
            hist,scores = cv.cross_validate(ret_results = True,verbose=1)
            
            results['Test accuracy']    .append( scores['Test accuracy'])
            results['Test sco']         .append( scores['Test sco'])
            results['Test loss']        .append( scores['Test loss'])
            results['Normal ECGs score'].append( scores['Normal ECGs score'])
            results['AF ECGs score']    .append( scores['AF ECGs score'])
            results['Other ECGS score'] .append( scores['Other ECGS score'])
            results['Noisy ECGS score'] .append( scores['Noisy ECGS score'])
            results['Total Score']      .append( scores['Total Score'])



            pd.DataFrame(results).to_csv(name)