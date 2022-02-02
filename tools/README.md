# Utilerías

Herramientas para usos varios.

- **Aplicación para etiquetado de calidad para imágenes de fondo de ojo**. Esta aplicación de escritorio con una interfaz gráfica que permite realizar el etiquetado de imágenes de fondo de ojo en base a su calidad. Se puede realizar el control de visualización a través de un panel de botones para presentar imagen previa. También cuenta con un botón para etiquetado de la imagen que está siendo visualizada. Las etiquetas son archivos json que llevan como nombre el mismo que el de la imagen actual. La etiqueta también contienen la estructura de calidad de la imagen en cuestión.
- **Aplicación para creación de máscaras a partir de archivos con anotaciones**. Esta aplicación permite la creación e imágenes que representan la etiqueta de otro conjunto de imágenes de fondo de ojo. Su insumo son una serie de archivos (json) que contienen polígonos; estos contienen información de los elementos anatómicos como el disco óptico y la mácula.

## Instalación de ambiente

Creación del ambiente:

```bash
conda env create -f environment.yml
```

Activación del ambiente:

```bash
conda activate retquality
```

## Aplicación para etiquetado de calidad para imágenes de fondo de ojo

```bash
usage: qualitylabel.py [-h] [-e] [-s IMAGE_SIZE] -o OPTIONS_FILE

Retina Quality Labeler

optional arguments:
  -h, --help            show this help message and exit
  -e, --relabel-mode    Re-labeling mode
  -s IMAGE_SIZE, --image-size IMAGE_SIZE
                        Image size
  -o OPTIONS_FILE, --options-file OPTIONS_FILE
                        Options file (json).
```

#### Activar modo reetiquetado

> ```python qualitylabel.py --relabel-mode --options-file /ruta/de/archivo/opciones/options.json```

#### Configurar tamaño de despliegue

Ejemplo a 640x640

> ```python qualitylabel.py --image-size 640 --options-file /ruta/de/archivo/opciones/options.json```

## Aplicación para creación de máscaras a partir de archivos con anotaciones

## Uso

```bash
usage: maskcreator.py [-h] -a ANNOT_PATH -o OUTPUT_PATH

Crops Creator.

optional arguments:
  -h, --help            show this help message and exit
  -a ANNOT_PATH, --annot_path ANNOT_PATH
                        Annotations directory
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Output directory
```

#### Ejemplo

> python maskcreator.py --annot_path /alguna/ruta/de/anotaciones --output_path /alguna/ruta/para/guardar

#### Used now

python qualitylabel.py --relabel-mode --sort_start_int --image-size 800 --options-file options/qualityClasses.json

python qualitylabel.py --sort_start_int --image-size 800 --options-file options/qualityClasses.json

python tools/qualitylabel.py --sort_start_int --image-size 800 --options-file tools/options/qualityClasses.json