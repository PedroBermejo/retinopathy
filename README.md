# Retinopathy
Repository for retinopathy tesis development

```bash
conda env create -f <folder-name>/environment.yml

conda activate <env-name>

conda env remove -n retinopathy
```

# Albumentations

It will take images in --good-path and take a sample of --sample number. It will save the good sample and the images with albumentations applied.
Everything is done inside this repository folder.

```bash
python albumentations/apply.py --good-path 'retinopathy-dataset/test/good/' \
       --good-sample-path 'retinopathy-dataset/test/albu/good_sample/'  \
       --bad-blur-sample-path 'retinopathy-dataset/test/albu/bad_blur_sample/' \
       --bad-gauss-noise-sample-path 'retinopathy-dataset/test/albu/bad_gauss_noise_sample/' \
       --bad-random-fog-sample-path 'retinopathy-dataset/test/albu/bad_random_fog_sample/' \
       --sample '100'
```

# Create-folders 

It will create folders from --path-images, with train, val and test, inside it will separate good and bad quality images, based on --csv-path csv values.

```bash
python create-folders/generate.py --path-images 'retinopathy-dataset/dataset/' \
       --path-images-idrid 'retinopathy-dataset/idrid-crop/idrid_crop/' \
       --path-images-messidor 'retinopathy-dataset/messidor-crop/messidor_crop/' \
       --path-good-quality-train 'retinopathy-dataset/train/good/' \
       --path-bad-quality-train 'retinopathy-dataset/train/bad/' \
       --path-good-quality-val 'retinopathy-dataset/val/good/' \
       --path-bad-quality-val 'retinopathy-dataset/val/bad/' \
       --path-good-quality-test 'retinopathy-dataset/test/good/' \
       --path-bad-quality-test 'retinopathy-dataset/test/bad/' \
       --csv-path 'retinopathy-dataset/result.csv'
```
It will verify there are no duplicates across different folders

```bash
python create-folder/verify_no_duplicates.py --base-path 'retinopathy-dataset'
```

# Histogram

Main script: graph_images_only_json.py

# Lighning

It will train models using transfer learning, it supports inceptionV3, mobilenetV2, resnet50 and vgg19

```bash
python lightning/train.py --model inceptionV3 \
       --train-path 'retinopathy-dataset/train' \
       --val-path 'retinopathy-dataset/val' \
       --save-path 'retinopathy-dataset/board'
        
```

This will load model and get metrics using test folder

```bash
python lightning/metrics.py --model inceptionV3 \
       --test-path 'retinopathy-dataset/test' \
       --model-path 'retinopathy-dataset/boards/board_inceptionV3/default/0/checkpoints/model-inceptionV3-epoch=09.ckpt' 
        
```

Run tensorboard and view graphs


```bash
tensorboard --logdir retinopathy-dataset/board/
```

This will run predicts over a couple of models

```bash
python lightning/metrics.py --good-path 'retinopathy-dataset/test' \
       --blur-path 'retinopathy-dataset/test' \
       --gauss-path 'retinopathy-dataset/test' \
       --fog-path 'retinopathy-dataset/test' \
       --inceptionV3-model-path 'retinopathy-dataset/boards/board_inceptionV3/default/0/checkpoints/model-inceptionV3-epoch=09.ckpt' \
       --mobilenetV2-model-path 'retinopathy-dataset/boards/board_inceptionV3/default/0/checkpoints/model-inceptionV3-epoch=09.ckpt' \
       --resnet50-model-path 'retinopathy-dataset/boards/board_inceptionV3/default/0/checkpoints/model-inceptionV3-epoch=09.ckpt' \
       --vgg19-model-path 'retinopathy-dataset/boards/board_inceptionV3/default/0/checkpoints/model-inceptionV3-epoch=09.ckpt'
        
```

# Problems

Exception in lightning:
Fix: downgrade python and open-cv:
- python==3.7.13
- opencv-python-headless==4.1.2.30