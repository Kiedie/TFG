import csv
import random


def create_dict_from_csv(f):
    '''
    Lee un CSV y lo convierte a un diccionario
    '''
    reader = csv.reader(f)
    return {row[0]: row[1] for row in reader}


def split_in_classes(raw_dict):
    '''
    Al diccionario devuelto por la función anterior lo revierte de tal manera que
    las claves son ahoras las clases y los valores son arrays con los ejemplos asocaidos a dichas 
    clases
    '''
    # Saca todas las etiquetas
    labels_set = set([x for x in raw_dict.values()])
    # Revierte el diccionario: {'~': [], 'A': [], 'O': [], 'N': []}, elas claves son las etiquetas y los valores listas con los atributos
    reverted_dict = {x: [] for x in labels_set}
    # Rellena el diccionario anterior
    for key, val in raw_dict.items():
        reverted_dict[val].append(key)

    return reverted_dict

def create_cross_validation_sets(fname, n_sets=5):
    '''
    Función que dada una ruta y el número de folds para el cross validation separa
    en el número de folds que se desea de tal manera que se mantengan las proporciones,
    hace un CV con muestreo estratificado
    '''
    # Abre el documento
    f = open(fname, 'r')
    # Transforma en un diccionario
    records_dict = create_dict_from_csv(f)
    # El diccionario lo transforma en otro en donde las claves son las clases
    splitted_dict = split_in_classes(records_dict)
    # Creo una listas de listas, en total tenemos 5 listas ( que son las particiones del cv)
    folds_dict = [{} for i in range(n_sets)]
    max_length = 0

    # Para cada clave-valor del diccionario donde las claves son las clases y los valores listas 
    for key, value in splitted_dict.items():
        # Ordeno de manera aleatoria la lista
        random.shuffle(value)
        # Tomo el tamaño de la lista
        size = len(value)
        i = 0
        # Reparto los valores en los distintos conjuntos de folds_dict
        for j in range(size):
            folds_dict[i][value[j]] = key
            i += 1
            i %= n_sets

    return folds_dict
