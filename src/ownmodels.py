from tensorflow.keras import optimizers, layers,regularizers,models
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.activations import relu, sigmoid

def full_convolution(x, filters, kernel_size, **kwargs):
    """
    It performs a Full convolution operation on the given keras Tensor.

    Parameters
    ----------
    x : Keras.Tensor
        Input tensor of the full convolution.
    filters : int
        Number of filters of the full convolution.
    kernel_size : int
        Kernel size of the convolution.
    kwargs : dict
        Rest of the arguments, optional.

    Returns
    -------
    x : Keras.Tensor
        Output tensor.
    """
    # Do a full convolution. Return a keras Tensor
    x = layers.ZeroPadding1D(padding=kernel_size - 1)(x)
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, **kwargs)(x)
    return x

def check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation):
    """
    Auxiliar function for checking the input parameters of the models.

    Parameters
    ----------
    include_top : bool
        Boolean value to control if the classification module should be placed in the model.
    weights : str
        Route to the saved weight of the model.
    input_tensor : keras.Tensor
        Input tensor of the model.
    input_shape : tuple
        Tuple with the input shape of the model.
    classes : int
        Number of classes to predict with the model.
    classifier_activation : str
        "softmax" or None

    Returns
    -------
    inp : Keras.Tensor
        Input tensor.
    """
    if include_top:
        if not isinstance(classes, int):
            raise ValueError("'classes' must be an int value.")
        act = keras.activations.get(classifier_activation)
        if act not in {keras.activations.get('softmax'), keras.activations.get(None)}:
            raise ValueError("'classifier_activation' must be 'softmax' or None.")

    if weights is not None and not tf.io.gfile.exists(weights):
        raise ValueError("'weights' path does not exists: ", weights)

    # Determine input
    if input_tensor is None:
        if input_shape is not None:
            inp = layers.Input(shape=input_shape)
        else:
            raise ValueError("One of input_tensor or input_shape should not be None.")
    else:
        inp = input_tensor

    return inp


def modelo_1(input_shape, num_classes, drop_out = 0.4, filters = 128, kernel_size = 5,padding = 'same'):
        
        model = models.Sequential()
        
        model.add( layers.Conv1D( filters, kernel_size, padding='same', input_shape=input_shape ) )
        
        model.add( layers.Activation('relu') )
        model.add( layers.MaxPooling1D() )
        model.add( layers.Dropout(0.4) )
        
        for i in range(6):
            model.add( layers.Conv1D(int(filters/(i+1)),kernel_size+(2*i),padding=padding))
            model.add( layers.Activation('relu'))
            model.add( layers.MaxPooling1D())
            model.add( layers.Dropout(0.4))

        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dense(128))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(drop_out))
        model.add(layers.Dense(64))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(drop_out))
        model.add(layers.Dense(32))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(drop_out))
        model.add(layers.Dense(num_classes))
        model.add(layers.Activation('softmax'))

        return model


def modelo_2(input_shape, 
            num_classes,
            drop_out = 0.4, 
            kernel_size = 5, 
            lstm_units = 8, 
            dense_units = 8, 
            padding='same'):

    inp = layers.Input(shape=input_shape)
    
    #model = models.Sequential()

  
    x = full_convolution(inp, filters=128, kernel_size=5, activation=relu, use_bias=False, strides=1)
    x = layers.MaxPooling1D()(x)
    x = full_convolution(x, filters=64, kernel_size=5, activation=relu, use_bias=False, strides=1)
    x = layers.MaxPooling1D()(x)
    x = full_convolution(x, filters=32, kernel_size=5, activation=relu, use_bias=False, strides=1)
    x = layers.MaxPooling1D()(x)
    x = full_convolution(x, filters=8, kernel_size=5, activation=relu, use_bias=False, strides=1)
    x = layers.MaxPooling1D()(x)

    x = layers.LSTM(units = lstm_units, recurrent_dropout = 0.2)(x)
    x = layers.Dropout(drop_out)(x)
    
    x = layers.Dense(units = dense_units)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(drop_out)(x)

    x =layers.Dense(num_classes)(x)
    x =layers.Activation('softmax')(x)

    model = keras.Model(inputs=inp,outputs=x,name='modelo2')

    return model



def modelo_3(input_shape, 
            num_classes, 
            dropout = 0.4, 
            return_sequences = False,
            gru_units = 8,
            dense_units= 16,
            padding = 'valid'):

    inp = layers.Input(shape=input_shape)


    x = full_convolution(inp, filters=128, kernel_size=5, activation=relu, use_bias=False, strides=1)
    x = layers.MaxPooling1D()(x)
    x = full_convolution(x, filters=64, kernel_size=5, activation=relu, use_bias=False, strides=1)
    x = layers.MaxPooling1D()(x)
    x = full_convolution(x, filters=32, kernel_size=5, activation=relu, use_bias=False, strides=1)
    x = layers.MaxPooling1D()(x)
    x = full_convolution(x, filters=8, kernel_size=5, activation=relu, use_bias=False, strides=1)
    x = layers.MaxPooling1D()(x)    



    x = layers.GRU(units=gru_units, dropout = dropout)(x)
    
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(units=dense_units)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x) 
    x = layers.Dense(units=num_classes)(x)
    x = layers.Activation('softmax')(x)

    model = keras.Model(inputs=inp,outputs=x,name='modelo3')

    
    return model



def modelo_4(input_shape,lstm_units, num_classes, drop_out = 0.4, filters = 128, kernel_size = 5,padding = 'same'):
        
        model = models.Sequential()
        
        model.add( layers.Conv1D( filters, kernel_size, padding='same', input_shape=input_shape ) )
        
        model.add( layers.Activation('relu') )
        model.add( layers.MaxPooling1D() )
        model.add( layers.Dropout(0.4) )
        
        for i in range(6):
            model.add( layers.Conv1D(int(filters/(i+1)),kernel_size+(2*i),padding=padding))
            model.add( layers.Activation('relu'))
            model.add( layers.MaxPooling1D())
            model.add( layers.Dropout(0.4))


        model.add(layers.LSTM(units = lstm_units,recurrent_dropout = 0.2, return_sequences=True))
        model.add(layers.LSTM(units = lstm_units,recurrent_dropout = 0.2, return_sequences=False))
    
        model.add(layers.Dense(128))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(drop_out))
        model.add(layers.Dense(64))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(drop_out))
        model.add(layers.Dense(32))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(drop_out))
        model.add(layers.Dense(num_classes))
        model.add(layers.Activation('softmax'))

        return model