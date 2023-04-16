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
python albumentations/apply.py --good-path 'retinopathy-dataset/albu_images/good_sample/' \
       --bad-blur-sample-path 'retinopathy-dataset/albu_images/bad_blur_sample/' \
       --bad-gauss-noise-sample-path 'retinopathy-dataset/albu_images/bad_gauss_noise_sample/' \
       --bad-random-fog-sample-path 'retinopathy-dataset/albu_images/bad_random_fog_sample/'
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

The following will first clear folders inside 'ablu_images', then it will subtract images used to train the models from the dataset with all images, it will just consider good quality images:

```bash
python create-folders/goodImages.py --path-all-images 'retinopathy-dataset/data_3_labels_idrid_messidor/' \
       --path-used-images 'retinopathy-dataset/data_3_labels_idrid_messidor_balanced/'  \
       --path-save-difference 'retinopathy-dataset/albu_images/' 
```

# Histogram

Main script: graph_images_only_json.py

This will create box plots over the softmax probabilities.

```bash
python histogram/boxplot.py --path-predicts 'retinopathy-dataset/boards/predicts.csv'      
```

This will read all labels in json files linked to each image in the kaggle dataset and count the numbers 
for the labels

```bash
python histogram/graph_csv_rLabel.py --path-datasets 'retinopathy-dataset/data_3_labels'      
```


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
tensorboard --logdir retinopathy-dataset/lightning_logs/
```

This will run predicts over a couple of models

```bash
python lightning/predicts.py --good-path 'retinopathy-dataset/albu_images/good_sample' \
       --blur-path 'retinopathy-dataset/albu_images/bad_blur_sample_crop' \
       --gauss-path 'retinopathy-dataset/albu_images/bad_gauss_noise_sample_crop' \
       --fog-path 'retinopathy-dataset/albu_images/bad_random_fog_sample_crop' \
       --inceptionV3-model-path 'retinopathy-dataset/boards/checkpoints_5/model-inceptionV3-epoch=11.ckpt' \
       --mobilenetV2-model-path 'retinopathy-dataset/boards/checkpoints_1/model-mobilenetV2-epoch=03.ckpt' \
       --resnet50-model-path 'retinopathy-dataset/boards/checkpoints_2/model-resnet50-epoch=13.ckpt' \
       --vgg19-model-path 'retinopathy-dataset/boards/checkpoints_3/model-vgg19-epoch=03.ckpt' \
       --path-to-csv 'retinopathy-dataset/boards/predicts'
        
```

This will format predicts csv

```bash
python lightning/formatPredicts.py --inception-predicts 'retinopathy-dataset/boards/predictsinception.csv' \
       --mobilenet-predicts 'retinopathy-dataset/boards/predictsmobilenet.csv' \
       --resnet-predicts 'retinopathy-dataset/boards/predictsresnet.csv' \
       --vgg-predicts 'retinopathy-dataset/boards/predictsvgg.csv' \
       --output 'retinopathy-dataset/boards/predicts.csv' 
        
```

# Cropper

It will remove background noise, attempting to make completely black background

```bash
python crop/run.py --model_path 'retinopathy-dataset/albu_images/segmenter.ckpt' \
       --src 'retinopathy-dataset/albu_images/bad_blur_sample/,retinopathy-dataset/albu_images/bad_gauss_noise_sample/,retinopathy-dataset/albu_images/bad_random_fog_sample/' \
       --dst 'retinopathy-dataset/albu_images/bad_blur_sample_crop/,retinopathy-dataset/albu_images/bad_gauss_noise_sample_crop/,retinopathy-dataset/albu_images/bad_random_fog_sample_crop/' 
```

# Problems

Exception in lightning:
Fix: downgrade python and open-cv:
- python==3.7.13
- opencv-python-headless==4.1.2.30