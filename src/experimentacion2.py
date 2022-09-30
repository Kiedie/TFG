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
name_modelo  = 'modelo2_new'
csv_name     = 'modelo2_new'+str(experimento)

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
           'filters':[128,256],
           'lstm_units': [8,16,32],
           'padding':['same']}

name = 'modelo_new.csv'


results = {
            'lstm_units':[],
            'filters':[],
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
    for filt in tuning['filters']:
        for padding in tuning['padding']:
                
            count +=1

            results['lstm_units'].append(lstm)
            results['filters'].append(filt)
                

            model = modelo_4( input_shape = dict_parameters['input_shape'], 
              num_classes = dict_parameters['classes'],
              lstm_units = lstm, 
              filters = filt,
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
                
                
            results['Test accuracy']    .append( scores['Test accuracy'])
            results['Test sco']         .append( scores['Test sco'])
            results['Test loss']        .append( scores['Test loss'])
            results['Normal ECGs score'].append( scores['Normal ECGs score'])
            results['AF ECGs score']    .append( scores['AF ECGs score'])
            results['Other ECGS score'] .append( scores['Other ECGS score'])
            results['Noisy ECGS score'] .append( scores['Noisy ECGS score'])
            results['Total Score']      .append( scores['Total Score'])
                


                
            pd.DataFrame(results).to_csv(name)
            
