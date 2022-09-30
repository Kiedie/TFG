from __future__ import division
import numpy as np
import pandas as pd
import sklearn.metrics as mt
import tensorflow as tf
import tensorflow.keras.backend as K
import math
import sys
from utils import sco

from data import DataGenerator
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras import optimizers, callbacks, Input, layers, Model
from sklearn.metrics import confusion_matrix
from tensorflow import keras

np.random.seed(42)

class CrossValidation:
    
    
    
    def __init__(self, X, y, 
                 model, model_name, 
                 num_classes = 4, n_split = 5 ,
                 epochs = 500,
                 csv_name = None,
                 opt = optimizers.RMSprop(),
                 batch_size = 128,
                 metricas = ['accuracy',sco],
                 CheckPoint = False, EarlyStopping = False,
                 patiente   = 50,
                 using_generators = True,
                 tensorboard = False):
        
        # Conjunto de datos completo y su etiqueta 
        self.X                = X
        self.y                = y
        self.seed   = 42
        # Data
        self.using_generators = using_generators
        
        ##
        self.X_train = None
        self.y_train = None
        self.X_val   = None
        self.y_val   = None
        self.X_test  = None
        self.y_test  = None
        
        self.train_generator = None
        self.val_generator   = None
        
        self.n_split = n_split
        
        # for model
        self.model            = model
        self.model_name       = model_name
        self.num_classes      = num_classes
        
        # for callbacks
        self.CheckPoint       = CheckPoint
        self.count_checkpoint = 0
        self.EarlyStopping    = EarlyStopping
        self.patiente         = patiente
        self.csv_name         = csv_name
        self.tensorboard      = tensorboard
        
        # For cross_validation - No incluido loss_function porque el problema solo va a requerir de uno 
        self.batch_size = batch_size
        self.opt        = opt
        self.metricas   = metricas
        self.epochs     = epochs

    
    # Función usadsa para testing
    def _show_datasets(self):
        print("X_train: ", self.X_train)
        print("y_train: ", self.y_train)
        print("X_test: ", self.X_test)
        print("y_test: ", self.y_test)
        print("X_val: ", self.X_val)
        print("y_val: ", self.y_val)
        

    
    def _check_sets(self):
        """
        Checks internal state
        
        Params:
        ----------
        generator: bool, default = True
        """
        assert(self.X_train != None), "X_train is None in class CrossValidation"
        assert(self.X_test  != None), "X_test  is None in class CrossValidation"
        assert(self.X_val   != None), "X_val   is None in class CrossValidation"
        assert(self.y_train != None), "y_train is None in class CrossValidation"
        assert(self.y_test  != None), "y_test  is None in class CrossValidation"
        assert(self.y_val   != None), "y_val   is None in class CrossValidation"
        
        if self.using_generators == True:
            assert(self.train_generator != None), "train_generator is None in class CrossValidation"
            assert(self.val_generator   != None), "val_generator is None in class CrossValidation"
            

    
    def _update_train_val_generator(self, train_generator = None, val_generator = None):
        """
        Update train and/or val generators.
        """
        
        assert(train_generator != None or val_generator != None), "train_generator and val_generator are both None"
        
        if train_generator != None:
            self.train_generator = train_generator
            
        if val_generator != None:
            self.val_generator = val_generator
    
    
    def _update_train_val_test(self, x_train,y_train,x_val,y_val,x_test,y_test):
        """
        update training, validation and test set
        """
        
        self.X_train = x_train
        self.y_train = y_train
        
        self.X_val   = x_val
        self.y_val   = y_val
        
        self.X_test  = x_test
        self.y_test  = y_test
        
        
    
    def _callbacks(self,
                   arg_es = {'monitor':'val_loss', 'mode':'auto','restore_best_weights':True} ,
                   arg_cp = {'monitor':'val_loss', 'verbose':1  , 'mode':'min','period':15}):
        """
        Instantiation of callbacks. It can generate early stopping or/and checkpointer
        
        Params:
        ----------
            arg_es:        dict, default = None
                Arguments for Early Stopping, it is a dictionary whose values are {monitor, paciencia,mode}
            
            arg_cp:        dict, dafault = None
                Arguments for Check Pointer, it is a dictioanry whose values are 
                {monitor,verbose,mode}
            
        """
        callback = []
        
        if self.EarlyStopping:
            earlyStopping = callbacks.EarlyStopping(
                    monitor  = arg_es['monitor'],
                    #mode     = arg_es['mode']
                    patience = self.patiente)
            callback.append(earlyStopping)
            
        if self.CheckPoint:
            checkpointer  = callbacks.ModelCheckpoint(
                filepath       = f'tmp/checkpoint/{self.model_name}_{self.count_checkpoint}',
                monitor        = arg_cp['monitor'], 
                verbose        = arg_cp['verbose'],
                save_best_only = True,
                #mode           = arg_cp['mode'], # En caso de duda consultar documentación
                save_freq      = arg_cp['period'])
            callback.append(checkpointer)
            
        if len(callback) < 1:
             raise ValueError("_callbacks in class CrossValidation function has been called but it returns nothing")
        
        return callback
    
    
    def _select_model(dict_parameters): # NO ESTA TESTEADO Y DE MOMENTO NO LO USAMOS
        """
        Select the model and its params
        
        Note:
        ----------
        An alternative to this would be select the model before instantiation of this class object.
        
        Params:
        ----------
            dict_parameters: dict
                A dicctionary containing the parameters of the model
        """
        
        if self.model_name == 'OhShuLih':
            model = OhShuLih(include_top            = dict_parameters['include_top'],
                             weights                = dict_parameters['weights'],
                             input_tensor           = dict_parameters['input_tensor'],
                             input_shape            = dict_parameters['input_shape'],
                             classes                = dict_parameters['classes'],
                             classifier_activations = dict_parameters['classifier_activations'])
            
        
            return model
        
        
        
    def _compile_model(self, loss_function = keras.losses.categorical_crossentropy):
        
        """
        Compile the model
        
        Params:
        ----------
            loss_funtion: keras.loss, default = categorical_crossentropy
        """
        
        self.model.compile(optimizer = self.opt, loss = loss_function, metrics = self.metricas)
    
    
    
    def _fit_model( self,
                    verbose       = 1,
                    arg_es        = {'monitor':'val_loss', 'patience':50, 'mode':'min'},
                    arg_cp        = {'monitor':'val_loss', 'verbose':1  , 'mode':'min','period':15}):
        
        """
        Fit the model. 
        Note that you can indicate if you want to use generators or not during training 
        by setting the parameter 'generator' to true or false. This function transforms numerical labels into categorical.
        
        Params:
        ----------
            generator: bool, default = True
                It means if we want to use or not train and val generator or not
            
            epchos: int, default = 500
            
            batch_size: int, default = 128
            
            verbose: {0,1}, default = 1
            
            arg_es: dict, default = None
                Dict containing the params of earlyStopping
                
            arg_cp: dict, default = None
                Dict containing the params of CheckPoint
                
        Return:
        ----------
            history: ??
        """
        
        # Declaramos los parámetros a usar en función de si usamos generadores o no
        if self.using_generators:
            #self._check_sets(generator)
            y_param = None
            x_param = self.train_generator
            val     = self.val_generator
        else:
            #self._check_sets(generator)
            y_param = tf.keras.utils.to_categorical(self.y_train,self.num_classes)
            x_param = self.X_train
            val     = (self.X_val, tf.keras.utils.to_categorical(self.y_val,self.num_classes))
        
        
        # Callbacks
        callbacks = None
        if self.EarlyStopping or self.CheckPoint:
            callbacks = self._callbacks(arg_es,arg_cp)
        
        # Activar funcionalidad de tensorboard
        if self.tensorboard:
            log_dir = 'logs'+self.csv_name
            callbacks = tf.keras.callbacks.TensorBoard(log_dir= log_dir,histogram_freq=1)


        # Entrenar el modelo          
        history = self.model.fit(
            x                =  x_param,
            y                =  y_param,
            batch_size       =  self.batch_size,
            epochs           =  self.epochs,
            verbose          =  verbose,
            steps_per_epoch  =  round(len(self.y_train)/self.batch_size),
            validation_data  =  val,
            validation_steps =  round(len(self.y_val)/self.batch_size),
            callbacks        =  callbacks
        )
        
        return history
    
    
    def print_model(self):
        """
        Display the model
        """
        print(self.model.summary())
        
     
    def _train_val_split(self,train_index):
        
        """
        Specify the number of elements should be include in training and val
        Args:
            train_index (int): Training index obtained by split in cross validation

        Returns:
            tuple of int: (number of samples for validation - number of samples for training)
        """
        index = train_index.copy()
        np.random.shuffle(train_index)
        nval   = round(len(train_index)*0.2)
        ntrain = round(len(train_index)*0.8)
        assert(nval+ntrain == len(train_index)), "Error in Cross Validation: nval+ntrain!=train_index"
    
        return (nval,ntrain)
        
    
    def _get_sets_from_index(self,train_index,test_index):
        
        """"
        Make data partitions in training, validation and test from index
        
        Args:
            train_index (int) :  train index obtained by split in cross validation
            test_index  (int) :  test index obtained by split in cross validation
            
        Returns: 
            dictionary: A dictionary containing the data and labels of validation, training and test
        
        """
        
        X_val_train , X_test  = self.X[train_index], self.X[test_index]
        y_val_train , y_test  = self.y[train_index], self.y[test_index]
        
        nval, ntrain = self._train_val_split(train_index)
        
        X_train, X_val = X_val_train[:ntrain], X_val_train[ntrain:ntrain+nval]
        y_train, y_val = y_val_train[:ntrain], y_val_train[ntrain:ntrain+nval]
        
        ret = {'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test,'X_val':X_val,'y_val':y_val}
    
        
        return ret 
    
    
    def cross_validate(self,
                       verbose,
                       csv    = True,
                       show   = False,
                       ret_results = False,
                       keys   = ['Test loss', 'Test accuracy','Test sco','Normal ECGs score','AF ECGs score','Other ECGS score','Noisy ECGS score','Total Score'],
                       cols   = ['Normal','Fib. art','Other','Ruido'],
                       arg_es = {'monitor':'val_loss', 'patience':50, 'mode':'min'},
                       arg_cp = {'monitor':'val_loss', 'verbose':1  , 'mode':'min', 'period':15}):
    
        """
            It makes Cross Validation.
            
            Params:
            ----------
                generatur: Bool,
                    Indicate if we use class DataGenerator or not
        """
            
        # Descomentar la línea de abajo para enviar el flujo de datos de salida a un fichero con el nombre del modelo 
        #sys.stdout = open(f'{self.model_name}.txt', "w")
        
    
        result_buffer     = [] # Guarda los resultados de las metricas 
        y_pred_buffer     = [] # Guarda matrices (test_samples,classes) que contienen probabilidades
        
        # Guardamos los pesos
        weights = self.model.get_weights()
        
        ## Si hacemos varios folds
        if self.n_split > 1:
            
            first_time  = True 
            n_fix       = 0
            hist        = []
            
            skf = StratifiedKFold(n_splits = self.n_split, shuffle = True, random_state = self.seed)
            for train_index, test_index in skf.split(self.X,self.y):
                
                # Suele ocurrir que si al hacer la partición, haya algunos folds que tengan 
                # una unidad más o menos en el conjunto de entrenamiento y test
                # en caso de que la división no sea exacta, por ejemplo
                # si tenemos 8528 samples y hacemos 5 folds
                # Van a ver tres folds en el que la longitud de test_index sea 1706 y dos en donde sea 1705
                # Esto es un problema a la hora de hacer la matriz de confusión ya que deben de ser todas iguales
                # Por tanto lo que vamos a hacer es normalizar esto y forzar a que siempre sea lo mismo
                
                # Si es la primera vez que entramos en el bucle
                # Guardmos la cantidad fija que vamos a usar como samples en test
                if first_time:
                    first_time = False
                    n_fix      = len(test_index)
                # Si ya no estamos en el primer fold
                # Vemos si concide, en caso contrario, forzamos a que sean iguales
                else:
                    # Si tiene uno por encima
                    if n_fix == (len(test_index) - 1):
                        # Quitamos un sample a test y lo añadimos a train
                        index = test_index[0]
                        test_index = test_index[1:]
                        train_index = np.append(train_index,index)
                    # Si está uno por debajo
                    elif n_fix == (len(test_index) + 1):
                        # Quitamos un sample a 
                        index = train_index[0]
                        train_index = train_index[1:]
                        test_index = np.append(test_index,index)
                        
                # NOTA: Con lo anterior no se está haciendo DataSnooping. No sea insensato y piense por un momento. Como no tenemos datos de test.
                # Lo que hacemos en cada fold es seleccionar un conjunto de train, val y test. Entrenamos, usamos validacion para seleccionar los 
                # pesos correctos con EarlyStopping y luego evaluamos en test y son esos resultados lo que usamos.
                
                
                

                # Obtenemos los conjuntos de validation, entrenamiento y test en foma de diccionario
                conjuntos = self._get_sets_from_index(train_index,test_index)
                
                # Actualizamos el estado interno con los conjuntos anteriormente obtenidos
                self._update_train_val_test( x_train = conjuntos['X_train'], y_train = conjuntos['y_train'], 
                                            x_val   = conjuntos['X_val'],   y_val   = conjuntos['y_val'], 
                                            x_test  = conjuntos['X_test'],  y_test  = conjuntos['y_test'])
                
                # Instanciamos generadores y actualizamos el estado interno 
                self._update_train_val_generator(train_generator = DataGenerator(self.X_train, self.y_train, batch_size = self.batch_size),
                                                 val_generator   = DataGenerator(self.X_val,   self.y_val,   batch_size = self.batch_size, subset='validation'))
                
                # Compilamos el modelo con la función de perdida por defecto categorical cross entropy
                self._compile_model()
                
                # Entrenamos el modelo, el manejo de callbacks se incluyen dentro de la función de abajo
                histo = self._fit_model(verbose, arg_es, arg_cp)
                hist.append(histo)
                
                # Evaluamos el modelo  
                scores = self._evaluate_model(verbose)
                
                # Realizamos las predicciones, obtenemos las probabilidades y la etiqueta que asigna respectivamente
                y_pred, final_pred = self._predict_model()
                
                # Obtenemos los resultados de las predicciones y la matriz de confusion en un diccionario
                results_dict = self.get_results(final_pred)
                
                
                # Guardamos las probabilidades de las predicciones finales y el diccionario de los resultados en un buffer 
                y_pred_buffer.append(y_pred)
                result_buffer.append(results_dict)
                
                
                # Restauramos los pesos
                self.model.set_weights(weights)
                
            
            # Si queremos ejecutarlo todo sin folds
        else:
            index = np.arange(self.X.shape[0])
            # Obtenemos indices y desordenamos
            np.random.shuffle(index)
            # Calculamos el número de ejemplos que vamos a usar para train, val y test
            ntest   = round(len(index)*0.2)
            nval    = round((len(index)-ntest)*0.2)
            ntrain  = round((len(index)-ntest)*0.8)
            assert(nval+ntrain+ntest == len(index)), "Error in Cross Validation: nval+ntrain!=train_index"
                                            

            # Tomamos indices
            train_index = index[:ntrain]
            val_index   = index[ntrain:ntrain+nval]
            test_index  = index[ntrain+nval:ntrain+nval+ntest]
            # Obtenemos subcobjuntos de train,test,val
            train  = self.X[train_index]

            ytrain = self.y[train_index]
            val    = self.X[val_index]
            yval   = self.y[val_index]

            test   = self.X[test_index]
            ytest  = self.y[test_index]
                       
            # Actualizamos el estado interno con los conjuntos anteriormente obtenidos
            self._update_train_val_test( x_train = train, y_train = ytrain, 
                                         x_val   = val  , y_val   = yval, 
                                         x_test  = test , y_test  = ytest)
                            
            # Instanciamos generadores y actualizamos el estado interno 
            self._update_train_val_generator(train_generator = DataGenerator(self.X_train, self.y_train, batch_size = self.batch_size),
                                            val_generator   = DataGenerator(self.X_val,   self.y_val,   batch_size = self.batch_size, subset='validation'))
                            
            # Compilamos el modelo con la función de perdida por defecto categorical cross entropy
            self._compile_model()
                                
            # Entrenamos el modelo, el manejo de callbacks se incluyen dentro de la función de abajo
            hist = self._fit_model(verbose, arg_es, arg_cp)
                            
            # Evaluamos el modelo  
            scores = self._evaluate_model(verbose)
                            
            # Realizamos las predicciones 
            y_pred, final_pred = self._predict_model()
                            
            # Obtenemos los resultados de las predicciones y la matriz de confusion
            results_dict = self.get_results(final_pred)
            conf_mat     = self.get_confusion_matrix(final_pred)
                            
            final_pred_buffer.append(final_pred)
            result_buffer.append(results_dict)
            
            
            if csv:
                df_conf_mat = self.get_dataframe_from_conf_mat(conf_mat,cols)
                if self.csv_name == None:
                    path =  f'./conf_mat_{self.model_name}_{self.count_checkpoint}.csv'
                else:
                    path =  f'./conf_mat_{self.csv_name}.csv'
                self.dataframe_to_csv(df_conf_mat, path = path)
            
            # Restauramos los pesos
            
                
        # Hacemos las medias de las métricas obtenidas en cada folds y lo sacamos en un diccionario        
        results = self._make_mean(result_buffer,keys)
        
        # En cada fold obtengo una matrix de dimension (test_samples,classes)
        # La almaceno en un array de tal manera que la dimensión nueva es (n_folds,test_samples,classes) 
        # Construyo un vector de zeros de dimension (test_samples,classes)
        # itero en shape[0] (n_folds) y sumo las matrices con dimension shape[1],shape[2]
        # finalmente divido por shape[0] para obtener la media
        
        y_pred_buffer = np.asarray(y_pred_buffer)
        
        y_pred_mean = np.zeros((y_pred_buffer.shape[1],y_pred_buffer.shape[2])) # Vector qe contendrá una matrix de medias provenientes de y_pred_buffer
        
        # iteramos en la dimension 0 de (n_folds,test_samples,classes)
        for i in range(y_pred_buffer.shape[0]):
            y_pred_mean = y_pred_mean + y_pred_buffer[i]
        y_pred_mean = y_pred_mean / y_pred_buffer.shape[0]
        
        # Obtengo la etiqueta que corresponde al argumento máximo por filas de la matrix de dimension (test_samples,classes)
        mean_final_pred = [i for i in np.argmax(y_pred_mean,axis=1)]
        # Obtengo matriz de confusion 
        conf_mat     = self.get_confusion_matrix(mean_final_pred)
        
        
        # Los resultaods los pasamos a dataframe y de ahí a csv
        
        if csv:
            # Metricas
            df_result   = self.get_dataframe_from_dict(results)
            self.dataframe_to_csv(df_result)
            # Matriz de confusion
            df_conf_mat = self.get_dataframe_from_conf_mat(conf_mat,cols)
            self.dataframe_to_csv(df_conf_mat, path = f'./conf_mat_{self.csv_name}.csv')
        
                
        if show:
            self.print_results(final_pred)
            
        self.count_checkpoint+=1
        self.count_checkpoint%=self.num_classes
        
        if ret_results:
            return hist, results
        
        return hist
      

    def _make_mean(self,v_dict,keys):
        """
        Make the mean of the scores. That function is used after training all folds in cross_validation

        Args:
            v_dict (list of dict): A list where each elements is a dictionary. It has the structure given by the output of _get_results
            keys   (list of string): A list with all keys of dict
                {'Test loss', 'Test accuracy','Test sco','Normal ECGs score','AF ECGs score','Other ECGS score','Noisy ECGS score','Total Score'}
        """
                

        
        test_loss     = []
        test_accuracy = []
        test_sco      = []
        tn            = []
        ta            = []
        to            = []
        t_noise       = []
        total         = []
        
        # Para cada diccionario obtengo sus elementos y los guardo
        for dic in v_dict:                      
            test_loss    .append(dic[keys[0]])
            test_accuracy.append(dic[keys[1]])
            test_sco     .append(dic[keys[2]])
            tn           .append(dic[keys[3]])
            ta           .append(dic[keys[4]])
            to           .append(dic[keys[5]])
            t_noise      .append(dic[keys[6]])
            total        .append(dic[keys[7]])
            
        # Creo otro diccionary con las claves y las medias 
        ret = { keys[0]:np.array(test_loss    ).mean(),
                keys[1]:np.array(test_accuracy).mean(),
                keys[2]:np.array(test_sco     ).mean(),
                keys[3]:np.array(tn           ).mean(),
                keys[4]:np.array(ta           ).mean(),
                keys[5]:np.array(to           ).mean(),
                keys[6]:np.array(t_noise      ).mean(),
                keys[7]:np.array(total        ).mean()}
        
        return ret 
    
    
    def _evaluate_model(self,verbose):
        y_test_cat = tf.keras.utils.to_categorical(self.y_test,self.num_classes)
        scores = self.model.evaluate(self.X_test,y_test_cat,verbose=verbose)
        return scores
    
    
    def _predict_model(self):
        y_pred = self.model.predict(self.X_test)
        final_pred = [i for i in np.argmax(y_pred,axis=1)] # Probar
        return y_pred, final_pred
    
    
    def get_confusion_matrix(self,final_pred):
        
        """
        Params:
        ----------
            final_pred: int np.ndarray
                The predicts of the model. It must be an array of int representing the class.
                The variable is the second output of _predict_model
        Return:
        ----------
            Confusion matrix: ndarray of shape (n_classes, n_classes)
                Confusion matrix whose i-th row and j-th colun entry indicates the number 
                of samples with true label being i-th class and predites label being j-th class
        """
        
        conf_mat = confusion_matrix(self.y_test,final_pred)
        return conf_mat
    
    # Testear
    def print_results(self,final_pred):
        
        conf_mat = self.get_confusion_matrix(final_pred)
        column_sums = np.sum(conf_mat, axis = 1)
        row_sums = np.sum(conf_mat, axis = 0)
        
        scores = self._evaluate_model(verbose=1)
        
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
        print('Confusion matrix:')
        print(conf_mat)
        
        
        print("Partial results:")
        t_n = 2*conf_mat[0,0] / (column_sums[0] + row_sums[0])
        t_a = 2*conf_mat[1,1] / (column_sums[1] + row_sums[1])
        t_o = 2*conf_mat[2,2] / (column_sums[2] + row_sums[2])
        t_noise = 2*conf_mat[3,3] / (column_sums[3] + row_sums[3])
        
        
        print("    Normal ECGs score: {}".format(t_n))
        print("    AF ECGs score: {}".format(t_a))
        print("    Other ECGs score: {}".format(t_o))
        print("    Noisy ECGs score: {}".format(t_noise))

        print("Total score: {}".format((t_n + t_a + t_o + t_noise)/4))
        
    # Testear    
    def get_dataframe_from_conf_mat(self, conf_mat, cols):
        df = pd.DataFrame(conf_mat, columns = cols, index = cols)
        return df 
    
    
    def get_results(self, final_pred):
        
        """
        Take metrics and join them into a dicctionary
        
        Args:
            final_pred (numpy.ndarray, int) 
                Predicciones made by function _predict_model. 
        
        Returns:
            dict: A dicctionary containing all metrics
        """
        
        conf_mat = self.get_confusion_matrix(final_pred)
        column_sums = np.sum(conf_mat, axis = 1)
        row_sums = np.sum(conf_mat, axis = 0)
        
        scores = self._evaluate_model(verbose=0)
        
        t_n         = 2*conf_mat[0,0] / (column_sums[0] + row_sums[0])
        t_a         = 2*conf_mat[1,1] / (column_sums[1] + row_sums[1])
        t_o         = 2*conf_mat[2,2] / (column_sums[2] + row_sums[2])
        t_noise     = 2*conf_mat[3,3] / (column_sums[3] + row_sums[3])
        total_score =  (t_n + t_a + t_o + t_noise)/4
        
        ret = {'Test loss':scores[0],
               'Test accuracy':scores[1],
               'Test sco':scores[2],
               'Normal ECGs score':t_n,
               'AF ECGs score':t_a,
               'Other ECGS score':t_o,
               'Noisy ECGS score':t_noise,
               'Total Score':total_score}
        
        return ret
    
    
    def get_dataframe_from_dict(self,results):
        df = pd.DataFrame(results.items())
        return df
    
    
    def dataframe_to_csv(self,df,path = None):
        if path == None:
            if self.csv_name == None:
                path = f'./results_{self.model_name}.csv'
            else:
                path = f'./results_{self.csv_name}.csv'
        df.to_csv(path_or_buf = path, decimal = ',')
        
        
    def mostrarEvolucion(hist):
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']
        plt.plot(loss)
        plt.plot(val_loss)
        plt.legend(['Training loss', 'Validation loss'])
        plt.show()

        acc = hist.history['accuracy']
        val_acc = hist.history['val_accuracy']
        plt.plot(acc)
        plt.plot(val_acc)
        plt.legend(['Training accuracy','Validation accuracy'])
        plt.show()