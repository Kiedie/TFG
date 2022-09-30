# Paquetes comunes
import numpy as np
import math

# Paquetes para DataReader
import csv
import os
import wfdb

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paquetes DataGenerator
import random
import tensorflow as tf
from tensorflow.keras import utils



# =========================================================================================================
# =========================================== DATA READER =================================================
# =========================================================================================================

class DataReader:
    """
    Clase diseñada para la lectura de datos
    
    Instance Atributes
    ----------
        data_dir: str
          The name of the directory where the data is
          
        label_correspondence: dict
          Dicctionary where relationates the label with a unique number
          
        samples: numpy.ndarray
          Matrix NxM where N is the the namber of samples and M is the number of caracteristics 
        
        labels: numpy.ndarray
          Array with the labels of samples
          
        reduce_dim: boolean, default=False
            If it is true, take all the signal and make padding at the beggining, else only take a piece of signal with length 1000.
            
    How it works:
    ----------
        data_dir ='dataset/data'            
        reader = DataReader(data_dir,...)
        X, y = reader.get_samples_labels()
    """
    
    seed = 42
    labels_work_1 = {'N': 0,'A': 1,'O': 2,'~': 3}
    
    def __init__(self,data_dir,label_correspondence=labels_work_1, reduce_dim = False,windows_size = 1000):
        
        """
        Constructor
        """
        self.windows_size      = windows_size
        self.data_dir = data_dir
        self.labels_correspondence = label_correspondence
        self.reduce_dim = reduce_dim
        self.samples, self.labels = self._load_data()
        self.windows = windows_size
        
        
    
    def get_samples_labels(self):
        return (self.samples,self.labels)
    
    def get_labels_correspondence(self):
        return self.labels_correspondence
    
    def get_data_dir(self):
        return self.data_dir
    
    def get_vectorized_labels(self):
        """
        One hot encoding. It returns a numpy.ndarray with vectorized labels
        """
        labels = self.labels.copy()
        num_classes = len(set(labels))
        return utils.to_categorical(labels,num_classes=num_classes)
        
    def _unify_length(self,data,max_length):
        
        """
        Data length unification using zero padding
            
        Parameters:
        ----------
            data: numpy.ndarray or list
                Matrix containing all the samples
            
            max_length: int
                The size of the longest sample in the set of samples
                
        Return:
        ----------
            All the samples with the same length, the type is the same
            that parameter data.
        """
        """
        ret  = [
                    np.concatenate
                    ( 
                        (
                            data[i],
                            np.zeros((max_length-len(data[i]),1))
                        ),
                        axis = 0
                    ) 
                for i in range(len(data))]
        """
        ret = pad_sequences(data,maxlen=max_length,dtype="float64",padding="pre",truncating="pre",value=0)
        
        return ret
    
    def _read_signals_from_csv_padding(self,csv_reader,dir):
        
        """
        Read all the signal from the csv passed by argument
        
        Parameters:
        ----------
            csv_reader: csv.reader
                open csv file (it's necessary to run csv.reader previously)
            
        """
        x_set = []
        y_set = []
        max_length = 0
        # Each row represents a signal
        for row in csv_reader: 
            # Reading the signal
            curr_signal = wfdb.rdrecord(os.path.join(dir,row[0])).p_signal
            # Change the size to a column vector
            curr_signal = curr_signal.reshape(-1,1)
            # Normalize
            curr_signal = (curr_signal - np.mean(curr_signal))/np.std(curr_signal)
            # Obtain the longest length
            curr_length = curr_signal.shape[0]
            if curr_length > max_length:
                max_length = curr_length
            # Store the signal and its labels
            #curr_signal.reshape(-1,1)
            x_set.append(curr_signal)
            y_set.append(self.labels_correspondence[row[1]])
            
        return [x_set,y_set,max_length]

    def _read_signals_from_csv_padding_norm2(self,csv_reader,dir):
        
        """
        Read all the signal from the csv passed by argument
        
        Parameters:
        ----------
            csv_reader: csv.reader
                open csv file (it's necessary to run csv.reader previously)
            
        """
        x_set = []
        y_set = []
        max_length = 0
        # Each row represents a signal
        for row in csv_reader: 
            # Reading the signal
            curr_signal = wfdb.rdrecord(os.path.join(dir,row[0])).p_signal
            # Change the size to a column vector
            curr_signal = curr_signal.reshape(-1,1)
            # Normalize
            # Obtain the longest length
            curr_length = curr_signal.shape[0]
            if curr_length > max_length:
                max_length = curr_length
            # Store the signal and its labels
            #curr_signal.reshape(-1,1)
            x_set.append(curr_signal)
            y_set.append(self.labels_correspondence[row[1]])
            
        # Normalizamos
        mean = np.mean(x_set)
        std  = np.std(x_set)

        for i in range(len(y_set)):
            x_set[i] = (x_set[i] - mean)/std
            
        return [x_set,y_set,max_length]
    
    
    def _read_signals_from_csv_reduce_norm2(self,csv_reader,dir):
        x_set = []
        y_set = []
        max_length = 0
        # Each row represents a signal
        for row in csv_reader: 
            # Reading the signal
            curr_signal = wfdb.rdrecord(os.path.join(dir,row[0])).p_signal
            # Change the size to a column vector
            curr_signal = curr_signal.reshape(-1,1)

            # Obtain de length
            length = curr_signal.shape[0]
            if length > self.windows_size:
                index = np.random.randint(length - self.windows_size)
                curr_signal = curr_signal[index:index+self.windows_size]
                # Store the signal and its labels
                #curr_signal.reshape(-1,1)
                x_set.append(curr_signal)
                y_set.append(self.labels_correspondence[row[1]])



        # Normalizamos
        mean = np.mean(x_set)
        std  = np.std(x_set)

        for i in range(len(y_set)):
            x_set[i] = (x_set[i] - mean)/std

        return [x_set,y_set] 
    
    
    def _read_signals_from_csv_reduce(self,csv_reader,dir):
        
        """
        Lee las señales del csv pasado como argumento. Aquellas con una longitud pequeña se omiten, las que 
        tengan una longitud suficientemente grande, se toma una porción aleatoria de la señal para reducir el tamaño
        
        Parameters:
        ----------
            csv_reader: csv.reader
                open csv file (it's necessary to run csv.reader previously)
            
        Return:
        ----------
            [x_set,y_set,max_length]: lista
                X_set: lista con las señales
                y_set: lista con las etiquetas
            
        """
        x_set = []
        y_set = []
        max_length = 0
        # Each row represents a signal
        for row in csv_reader: 
            # Reading the signal
            curr_signal = wfdb.rdrecord(os.path.join(dir,row[0])).p_signal
            # Change the size to a column vector
            curr_signal = curr_signal.reshape(-1,1)
            # Normalize
            curr_signal = (curr_signal - np.mean(curr_signal))/np.std(curr_signal)
            # Obtain de length
            length = curr_signal.shape[0]
            if length > self.windows_size:
                index = np.random.randint(length - self.windows_size)
                curr_signal = curr_signal[index:index+self.windows_size]
                # Store the signal and its labels
                #curr_signal.reshape(-1,1)
                x_set.append(curr_signal)
                y_set.append(self.labels_correspondence[row[1]])
            
        return [x_set,y_set]
     
    def _convert_format(self,labels,):
        """
        The format of data we need has a shape of (N, T, V) where:
            N is the number of segments (or instances), 
            T is the number of timesteps of each segment,
            V is the number of variables (1 in this case).
        """
        # El formato lo ponemos de potra así que no nos preocupamos
        # pero si en un futuro necesitamos cambiar el formato de los datos
        # usamos esta función y la llamamos en _load_data
        pass
        
    
    def _load_data(self,csv_file='REFERENCE.csv'):
        """
        Obtain all the data and labels from csv file
        
        Parameters:
        ----------
            csv_file: str, default=REFERENCE.csv
                A csv containing the signal and its labels
                The format is like that: 'A01110'-'O'
                Realize that the file only has the name of the specify signal
        """
        # Take the one file we have, open and read it
        file = open(os.path.join(self.data_dir,csv_file))
        reader = csv.reader(file)
        
        # Store the csv-file's elements
        if self.reduce_dim == True:
            samples, labels = self._read_signals_from_csv_reduce_norm2(reader,self.data_dir)
        else:
            samples, labels, maxLength = self._read_signals_from_csv_padding_norm2(reader,self.data_dir)
            samples = self._unify_length(samples, maxLength)
        
        
        # Usamos _convert_format si queremos tener un formato específico.
        
        return (np.asarray(samples),np.asarray(labels))
        
    def get_cross_validation_iterator(stratify = True, nfolds = 5):
        pass


## ESTA ES UNA VERSIÓN ANTERIOR A LA DE ARRIBA, LA UNICA DIFERENCIA ESTÁ EN QUE ESTÁ NO REDUCE LA DIMENSIÓN DE LAS SEÑALES
class DataReader2:
    """
    Clase diseñada para la lectura de datos
    
    Instance Atributes
    ----------
        data_dir: str
          The name of the directory where the data is
          
        label_correspondence: dict
          Dicctionary where relationates the label with a unique number
          
        samples: numpy.ndarray
          Matrix NxM where N is the the namber of samples and M is the number of caracteristics 
        
        labels: numpy.ndarray
          Array with the labels of samples
    """
    
    seed = 42
    labels_work_1 = {'N': 0,'A': 1,'O': 2,'~': 3}
        
    
    def __init__(self,data_dir,label_correspondence=labels_work_1, reduce_dim = False,windows_size = 1000):
        
        """
        Constructor
        """
        self.windows_size      = windows_size
        self.data_dir = data_dir
        self.labels_correspondence = label_correspondence
        self.reduce_dim = reduce_dim
        self.samples, self.labels = None, None#self._load_data()
        self.windows = windows_size
        
        
    
    def get_samples_labels(self):
        return (self.samples,self.labels)
    
    def get_labels_correspondence(self):
        return self.labels_correspondence
    
    def get_data_dir(self):
        return self.data_dir
    
    def get_vectorized_labels(self):
        """
        One hot encoding. It returns a numpy.ndarray with vectorized labels
        """
        labels = self.labels.copy()
        num_classes = len(set(labels))
        return utils.to_categorical(labels,num_classes=num_classes)
        
    def _unify_length(self,data,max_length):
        
        """
        Data length unification using zero padding
            
        Parameters:
        ----------
            data: numpy.ndarray or list
                Matrix containing all the samples
            
            max_length: int
                The size of the longest sample in the set of samples
                
        Return:
        ----------
            All the samples with the same length, the type is the same
            that parameter data.
        """
        """
        ret  = [
                    np.concatenate
                    ( 
                        (
                            data[i],
                            np.zeros((max_length-len(data[i]),1))
                        ),
                        axis = 0
                    ) 
                for i in range(len(data))]
        """
        ret = pad_sequences(data,maxlen=max_length,dtype="float64",padding="pre",truncating="pre",value=0)
        
        return ret
    
    def _read_signals_from_csv_padding(self,csv_reader,dir):
        
        """
        Read all the signal from the csv passed by argument
        
        Parameters:
        ----------
            csv_reader: csv.reader
                open csv file (it's necessary to run csv.reader previously)
            
        """
        x_set = []
        y_set = []
        max_length = 0
        # Each row represents a signal
        for row in csv_reader: 
            # Reading the signal
            curr_signal = wfdb.rdrecord(os.path.join(dir,row[0])).p_signal
            # Change the size to a column vector
            curr_signal = curr_signal.reshape(-1,1)
            # Normalize
            curr_signal = (curr_signal - np.mean(curr_signal))/np.std(curr_signal)
            # Obtain the longest length
            curr_length = curr_signal.shape[0]
            if curr_length > max_length:
                max_length = curr_length
            # Store the signal and its labels
            #curr_signal.reshape(-1,1)
            x_set.append(curr_signal)
            y_set.append(self.labels_correspondence[row[1]])
            
        return [x_set,y_set,max_length]

    
    def _read_signals_from_csv_padding_norm2(self,csv_reader,dir):
        
        """
        Read all the signal from the csv passed by argument
        
        Parameters:
        ----------
            csv_reader: csv.reader
                open csv file (it's necessary to run csv.reader previously)
            
        """
        x_set = []
        y_set = []
        max_length = 0
        # Each row represents a signal
        for row in csv_reader: 
            # Reading the signal
            curr_signal = wfdb.rdrecord(os.path.join(dir,row[0])).p_signal
            # Change the size to a column vector
            curr_signal = curr_signal.reshape(-1,1)
            # Normalize
            # Obtain the longest length
            curr_length = curr_signal.shape[0]
            if curr_length > max_length:
                max_length = curr_length
            # Store the signal and its labels
            #curr_signal.reshape(-1,1)
            x_set.append(curr_signal)
            y_set.append(self.labels_correspondence[row[1]])
            
        # Normalizamos
        mean = np.mean(x_set)
        std  = np.std(x_set)

        for i in range(len(y_set)):
            x_set[i] = (x_set[i] - mean)/std
            
        return [x_set,y_set,max_length]
    
    
    def _read_signals_from_csv_reduce_norm2(self,csv_reader,dir):
        x_set = []
        y_set = []
        max_length = 0
        # Each row represents a signal
        for row in csv_reader: 
            # Reading the signal
            curr_signal = wfdb.rdrecord(os.path.join(dir,row[0])).p_signal
            # Change the size to a column vector
            curr_signal = curr_signal.reshape(-1,1)

            # Obtain de length
            length = curr_signal.shape[0]
            if length > self.windows_size:
                index = np.random.randint(length - self.windows_size)
                curr_signal = curr_signal[index:index+self.windows_size]
                # Store the signal and its labels
                #curr_signal.reshape(-1,1)
                x_set.append(curr_signal)
                y_set.append(self.labels_correspondence[row[1]])



        # Normalizamos
        mean = np.mean(x_set)
        std  = np.std(x_set)

        for i in range(len(y_set)):
            x_set[i] = (x_set[i] - mean)/std

        return [x_set,y_set] 

    
    
    def _read_signals_from_csv_reduce(self,csv_reader,dir):
        
        """
        Lee las señales del csv pasado como argumento. Aquellas con una longitud pequeña se omiten, las que 
        tengan una longitud suficientemente grande, se toma una porción aleatoria de la señal para reducir el tamaño
        
        Parameters:
        ----------
            csv_reader: csv.reader
                open csv file (it's necessary to run csv.reader previously)
            
        Return:
        ----------
            [x_set,y_set,max_length]: lista
                X_set: lista con las señales
                y_set: lista con las etiquetas
            
        """
        x_set = []
        y_set = []
        max_length = 0
        # Each row represents a signal
        for row in csv_reader: 
            # Reading the signal
            curr_signal = wfdb.rdrecord(os.path.join(dir,row[0])).p_signal
            # Change the size to a column vector
            curr_signal = curr_signal.reshape(-1,1)
            # Normalize
            curr_signal = (curr_signal - np.mean(curr_signal))/np.std(curr_signal)
            # Obtain de length
            length = curr_signal.shape[0]
            if length > self.windows_size:
                index = np.random.randint(length - self.windows_size)
                curr_signal = curr_signal[index:index+self.windows_size]
                # Store the signal and its labels
                #curr_signal.reshape(-1,1)
                x_set.append(curr_signal)
                y_set.append(self.labels_correspondence[row[1]])
            
        return [x_set,y_set]
     
    def _convert_format(self,labels,):
        """
        The format of data we need has a shape of (N, T, V) where:
            N is the number of segments (or instances), 
            T is the number of timesteps of each segment,
            V is the number of variables (1 in this case).
        """
        # El formato lo ponemos de potra así que no nos preocupamos
        # pero si en un futuro necesitamos cambiar el formato de los datos
        # usamos esta función y la llamamos en _load_data
        pass
        
    
    def _load_data(self,csv_file='REFERENCE.csv'):
        """
        Obtain all the data and labels from csv file
        
        Parameters:
        ----------
            csv_file: str, default=REFERENCE.csv
                A csv containing the signal and its labels
                The format is like that: 'A01110'-'O'
                Realize that the file only has the name of the specify signal
        """
        # Take the one file we have, open and read it
        file = open(os.path.join(self.data_dir,csv_file))
        reader = csv.reader(file)
        
        # Store the csv-file's elements
        if self.reduce_dim == True:
            samples, labels = self._read_signals_from_csv_reduce(reader,self.data_dir)
        else:
            samples, labels, maxLength = self._read_signals_from_csv_padding(reader,self.data_dir)
            samples = self._unify_length(samples, maxLength)
        
        
        # Usamos _convert_format si queremos tener un formato específico.
        
        return (np.asarray(samples),np.asarray(labels))
        
    def get_cross_validation_iterator(stratify = True, nfolds = 5):
        pass

    

# =========================================================================================================
# ======================================== DATA GENERATOR =================================================
# =========================================================================================================

class DataGenerator(utils.Sequence):
    def __init__(self, x, y, batch_size=128,
                 subset='train', num_classes=4,
                 distortion_fact=0.1, distortion_prob=0.3):
        self.x = x
        self.y = y
        self.num_classes = num_classes
        self.dist_fact = distortion_fact
        self.dist_prob = distortion_prob
        self.batch_size = batch_size
        self.subset = subset
        self.sep_dict = None
        
        if subset == 'train':
            self.sep_dict = {key: [] for key in set(y)}
            for count, value in enumerate(self.x):
                self.sep_dict[self.y[count]].append(value)        
        elif subset != 'validation':
            raise ValueError("Unsupported subset type")
            
    def __len__(self):
        """
        Return the number of batches we have
        """
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    
    def on_epoch_end(self):
        pass
    
    def input_shape(self):
        return self.x.shape
    
    def sample_shape(self):
        return self.x[0].shape
    
    def amp_signal(self,signal):
        return random.uniform(1-self.dist_fact,1+self.dist_fact)*signal
    
    def vertical_shift(self,signal):
        return random.uniform(-self.dist_fact,self.dist_fact)+signal
    
    def horizontal_shift(self,signal):
        shift = random.randint(0,len(signal))
        return np.roll(signal,shift)
    
    def peaks_noise(self,signal):
        ret = signal.copy()
        # Obteain the max value and create a threshold
        max_val = ret.max()
        max_threshold = 0.9*max_val
        # Get and modify the indexes greater than the threshold
        affected_values = ret > max_threshold
        ret[affected_values] *= np.random.uniform(1*self.dist_fact,1+self.dist_fact)
        return ret
        
    def __getitem__(self,idx):
        """
        Inherited method which serves as iterator for batches.
        In this method, we try to make balanced batches by having the same number of samples in each batch.
        """
        if self.subset == 'train':
            batch_x = []
            batch_y = []
            class_index = 0
            
            # Iteramos tantas veces como tamaño tenga el batch
            for i in range(self.batch_size):
                # Obtenemos un indice aleatorio perteneciente a la clase 'class_index'
                curr_index = random.randint(0,len(self.sep_dict[class_index])-1)
                # Obtenemos la señal de dicho indice
                curr_signal = self.sep_dict[class_index][curr_index]
                
                # Aplicamos una transformación
                if random.uniform(0,1) > self.dist_prob:
                    curr_signa = self.amp_signal(curr_signal)
                    
                if random.uniform(0,1) > self.dist_prob:
                    curr_signa = self.vertical_shift(curr_signal)
                    
                if random.uniform(0,1) > self.dist_prob:
                    curr_signa = self.horizontal_shift(curr_signal)
                    
                if random.uniform(0,1) > self.dist_prob:
                    curr_signa = self.peaks_noise(curr_signal)
                    
                # Guardamos la señal y su etiqueta en los batches y pasamos a la siguiente clase
                batch_x.append(curr_signal)
                batch_y.append(class_index)
                class_index += 1
                class_index %= self.num_classes
                
            return (np.asarray(batch_x),self._vectorize_labels(batch_y))
            
        elif self.subset == 'validation':
            # En el conjunto de validacion no se aplica transformación alguna.
            # Los datos se devuelven en batches directamente
            batch_x = self.x[idx*self.batch_size : (idx+1)*self.batch_size]
            batch_y = self.y[idx*self.batch_size : (idx+1)*self.batch_size]
            
            return (np.asarray(batch_x),self._vectorize_labels(batch_y))
    
    def _vectorize_labels(self,labels):
        """
        One hot encoding. It returns a numpy.ndarray
        """
        return utils.to_categorical(labels,num_classes=self.num_classes)
    
    
    def return_set(self):
        """
        Return the whole dataset (data with labels) using the next format: (numpy.ndarray, numpy.ndarray)
        """
        return(np.asarray(self.x),self._vectorize_labels(self.y.copy()))