# Cargamos librerías
import numpy as np
from data import DataReader, DataGenerator
from utils import *
from CrossValidation import *
import sys
sys.path.append("S-TSFE-DL")
from TSFEDL.models_keras import *
from tensorflow.keras import optimizers


def WangKejun2(include_top=True,
              weights=None,
              input_tensor=None,
              input_shape=None,
              classes=5,
              classifier_activation="softmax"):
    """
    References
    ----------
        Wang, Kejun, Xiaoxia Qi, and Hongda Liu. "Photovoltaic power forecasting based LSTM-Convolutional Network."
        Energy 189 (2019): 116225.

    Parameters
    ----------
        include_top: bool, default=True
          Whether to include the fully-connected layer at the top of the network.

        weights: str, default=None
            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None
            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None
            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5
            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'
            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------
        model: keras.Model
            A `keras.Model` instance.
    """
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    x = layers.LSTM(units=8, return_sequences=True)(inp)
    x = layers.LSTM(units=16, return_sequences=True)(x)
    x = layers.Conv1D(filters=32, kernel_size=3, strides=1, activation=relu)(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, strides=1, activation=relu)(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

    
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.1)(x)
    x = layers.Dense(units=128, activation=relu)(x)
    x = layers.Dense(units=32, activation=relu)(x)
    x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="WangKejun")

    if weights is not None:
        model.load_weights(weights)

    return model


################################################################################
############################### HIPERPARÁMETROS ################################
################################################################################

# HIPERPARÁMETROS
batch_size   = 16
epochs       = 300
patiente     = 50

generator    = True
reduce       = True
windows_size = 8000

lr           = 0.001
opt          = optimizers.Adam(learning_rate = lr)
opt_name     = 'Adam'

# Nombre modelo y del csv a sacar
experimento  =  3
name_modelo  = 'WangKejun'
csv_name     = 'WangKejun_'+str(experimento)

notas        = "En el experimento uno los resultados eran malisimo, decrementamos la complejidad del modelo"
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
model = WangKejun2( include_top              = dict_parameters['include_top'],
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
                      patiente          = patiente,
                      using_generators  = generator)

# Cross Validation
hist = cv.cross_validate(verbose=1)

scores = cv._evaluate_model(verbose=1)
print("Scores: ", scores)
