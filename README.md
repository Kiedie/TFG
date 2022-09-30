# Trabajo Fin de Grado 

- Desarrollo de modelos de Deep Learning basado en redes neuronales recurrentes para clasificación de arritmias

Desarrollado por:  
- Juan José Herrera Aranda

Titulación:
- Doble Grado Matemáticas e Ingeniería Informática

Tutores:
- Francisco Herrera Triguero
- Julián Luengo Martín

Departamento: 
- Ciencias de la Computación e Inteligencia Artificial (DCSAI)

## Abstract
Development of deep learning models based on convolutional and recurrent neuronal network for classification of arritmia using ECG. After preprocessing the ECG signals, CNN is applied for features extraction, followed by RNN for temporal feature treatment. Furthermore, the created models are compared with models obtained and supported by the literature.

## Archivos

- *\*.sh*: Scripts para ejecutar los archivos en servidores GPU
- *CrossValidation.py*: Implementación de la clase crossValidation
- *data.py*: Implementación de las clases DataReader y DataGenerator
- *ownmodels.py*: Implementación modelos propios
- *utils.py*: Fichero con algunas utilidades necesaris en otros ficheros
- *execute,experimentación\*.py*: Ficheros que realizan el proceso de experimentación de los modelos propios
- *GaoJunLi,ChenChen,OhShuLi.py*: Ficheros que contienen implementación y ejecución de los modelos propuestos por la literatura
- *S-TSFE-DL* (https://github.com/ari-dasci/S-TSFE-DL) Repositorio externo auxiliar en el que nos hemos apoyado 

__Nota:__ No se han incluido los ficheros de las salidas con los resultados y los logs de las ejecuciones.


## Ejecución

La ejecución no será posible a menos de que disponga acceso a los servidores GPU de la universidad de Granada y un entorno configurado.
Partiendo de dicha base basta con:

1. Descargar la base de datos cuyo enlace se encuentra en la memoria del proyecto.
2. En un directorio dejar la carpeta con la base de datos y todos estos ficheros. (La estructura que tiene en este repositorio no es apta para la ejecución)
3. Ejecutar el siguiente comando en la terminal (previamente haciendo conexión ssh con el servidor GPU) donde salida.txt es el archivo que se creará para ver el progreso de la ejecución y fichero.sh el script con el fichero que se quiere ejecutar.

```!bash
sbatch -o salida.txt fichero.sh
```

Vemos a continuación la estructura de un scripts para la ejecución
```!bash
#!/bin/bash
  
#SBATCH --job-name signals
#SBATCH --partition dios
#SBATCH -c 4
#SBATCH --gres=gpu:1
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/naguiler/Environments/tf-pt # El entorno virtual es prestado de Nacho

python3 experimentacion.py 

mail -s "Proceso finalizado" juanjoha@correo.ugr.es <<< "El proceso ha finalizado"
```
