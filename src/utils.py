#from __future__ import division
import numpy as np
import sklearn.metrics as mt
import tensorflow.keras.backend as K

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.initializers import orthogonal, he_uniform
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
import sys
sys.path.append("S-TSFE-DL")
from TSFEDL.models_keras import *




def sco(y_true, y_pred):
    y_pred_classes = K.argmax(y_pred, axis=-1)
    y_true_classes = K.argmax(y_true, axis=-1)
    conf_mat = tf.math.confusion_matrix(
        y_true_classes,
        y_pred_classes,
        num_classes=4
    )

    column_sums = tf.reduce_sum(conf_mat, axis = 1)
    row_sums = tf.reduce_sum(conf_mat, axis = 0)

    t_n = tf.cast(2*conf_mat[0,0]/(column_sums[0] + row_sums[0]), tf.float32)
    t_a = tf.cast(2*conf_mat[1,1]/(column_sums[1] + row_sums[1]), tf.float32)
    t_o = tf.cast(2*conf_mat[2,2]/(column_sums[2] + row_sums[2]), tf.float32)
    t_noise = tf.cast(2*conf_mat[3,3]/(column_sums[3] + row_sums[3]), tf.float32)

    t_n = tf.where(tf.math.is_nan(t_n), tf.cast(0, tf.float32), t_n)
    t_a = tf.where(tf.math.is_nan(t_a), tf.cast(0, tf.float32), t_a)
    t_o = tf.where(tf.math.is_nan(t_o), tf.cast(0, tf.float32), t_o)
    t_noise = tf.where(tf.math.is_nan(t_noise), tf.cast(0, tf.float32), t_noise)

    return (t_n + t_a + t_o + t_noise) / 4

def print_params(dic):
    
    print("=======================================")
    print(f"{dic['name_modelo']} ( Experimento {dic['experimento']})")
    print("=======================================")
    print("Params:")
    for clave,valor in dic.items():
        
        if valor != None and type(valor) != list:
            print(f"{clave:<20}{valor:>10}")
    

def parametros():
    """
        generator    = False
        reduce       = True
        windows_size = 3000
        folds        = 1


        batch_size   = 256
        epochs       = 100
        patiente     = 40
        n_split      = 1

        lr           = 0.001
        opt          = optimizers.Adam(learning_rate = lr)



        experimento = "13avo Experimento"
        notas       = ""
        name_modelo = 'khanzulfiqar'
        csv_name    = '13avo Experimento'
    """
    gen = 'False'
    red = 'False'
    if generator:
        gen='True'
    if reduce:
        red = 'True'
    
    print(experimento)
    print(f'Notas: {notas}')
    print(f'Modelo: {name_modelo}')
    print(f'csv_name: {csv_name}')
    print("")
    
    print(f'Generacion de datos: {gen}')
    print(f'Reducción de datos: {red}')
    if reduce:
        print(f'     Windows size: {windows_size}')
    print("")
    
    print(f'batch_size: {batch_size}')
    print(f'epochs: {epochs}')
    print(f'patiente: {patiente}')
    print(f'n_split: {n_split}')
    print("")
    
    print(f'Optimizador: {opt}')
    print(f'Learning Rate: {lr}')



def check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation):
    """
        Auxiliar function for checking the input parameters of the models.
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





def init_modelos(dict_parameters):

    ohshulih = OhShuLih( include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    
    khanzu   = KhanZulfiqar( include_top              = dict_parameters['include_top'],
                             weights                  = dict_parameters['weights'],
                             input_tensor             = dict_parameters['input_tensor'],
                             input_shape              = dict_parameters['input_shape'],
                             classes                  = dict_parameters['classes'],
                             classifier_activation    = dict_parameters['classifier_activations'])
    
    
    zhengzhenyu = ZhengZhenyu(include_top              = dict_parameters['include_top'],
                             weights                  = dict_parameters['weights'],
                             input_tensor             = dict_parameters['input_tensor'],
                             input_shape              = dict_parameters['input_shape'],
                             classes                  = dict_parameters['classes'],
                             classifier_activation    = dict_parameters['classifier_activations'])
    
    
    wangkejun = WangKejun( include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    
    """
    chenchen = ChenChen( include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    """
    
    kimtaeyoung = KimTaeYoung(include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    
    genminxing = GenMinxing(include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    
    fujiangmeng = FuJiangmeng(include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    """
    shihaotian = ShiHaotian(include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    """
    
    huangmeiling = HuangMeiLing( include_top              = dict_parameters['include_top'],
                                 weights                  = dict_parameters['weights'],
                                 input_tensor             = dict_parameters['input_tensor'],
                                 input_shape              = dict_parameters['input_shape'],
                                 classes                  = dict_parameters['classes'],
                                 classifier_activation    = dict_parameters['classifier_activations'])
    
    
    
    lihohshu =  LihOhShu(include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    
    
    
    gaojunli = GaoJunLi(include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    
    
    weixiaoyan = WeiXiaoyan(include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    
    
    
    
    kongzhengmin = KongZhengmin(include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    
    
    
    
    caiwenjuan = CaiWenjuan(include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    
    
    kimmingu  = KimMinGu(include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    
    htetmyetlynn = HtetMyetLynn(include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    
    Zhangjin = ZhangJin(include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    
    yaoqihang = YaoQihang(include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    
    yibogao = YiboGao(include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    
    hongtan = HongTan(include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    
    
    
    ret = { 'OhShuLih'     :ohshulih,
            'KhanZulfiqar' :khanzu,
            'ZhengZhenyu'  :zhengzhenyu,
            'WangKejun'    :wangkejun,
            #'ChenChen'     :chenchen,
            'KimTaeYoung'  :kimtaeyoung,
            'GenMinxing'   :genminxing,
            'FuJiangmeng'  :fujiangmeng,
            #'ShiHaotian'   :shihaotian,
            'HuangMeiLing' :huangmeiling,
            'LihOhShu'     :lihohshu,
            'GaoJunLi'     :gaojunli,
            'WeiXiaoyan'   :weixiaoyan,
            'KongZhengmin' :kongzhengmin,
            'CaiWenjuan'   :caiwenjuan,
            'KimMinGu'     :kimmingu,
            'HtetMyetLynn' :htetmyetlynn,
            'ZhangJin':Zhangjin,
            'YaoQihang':yaoqihang,
            'YiboGao':yibogao,
            'HongTan':hongtan,
          }
    
    return ret



def init_kim_gen_fuji(dict_parameters):


    kimtaeyoung = KimTaeYoung(include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    
    genminxing = GenMinxing(include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    
    fujiangmeng = FuJiangmeng(include_top              = dict_parameters['include_top'],
                         weights                  = dict_parameters['weights'],
                         input_tensor             = dict_parameters['input_tensor'],
                         input_shape              = dict_parameters['input_shape'],
                         classes                  = dict_parameters['classes'],
                         classifier_activation    = dict_parameters['classifier_activations'])
    
    dic = {'kimtaeyoung':kimtaeyoung,'genminxing':genminxing,'fujiangmeng':fujiangmeng}
    
    return dic 