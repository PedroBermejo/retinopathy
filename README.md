# Retinopathy
Repository for retinopathy tesis development 

# Albumentations

It will take images in --good-path and take a sample of --sample number. It will save the good sample and the images with albumentations applied.
Everything is done inside this repository folder.

```bash
python albumentations/apply.py --good-path 'retinopathy-dataset/test/good/' \\\
       --good-sample-path 'retinopathy-dataset/test/good_sample/'  \\\
       --bad-blur-sample-path 'retinopathy-dataset/test/bad_blur_sample/' \\\
       --bad-gauss-noise-sample-path 'retinopathy-dataset/test/bad_gauss_noise_sample/' \\\
       --bad-random-fog-sample-path 'retinopathy-dataset/test/bad_random_fog_sample/' \\\
       --sample '100'
```
# Create-folders 

It will create folders from --path-images, with train, val and test, inside it will separate good and bad quality images, based on --csv-path csv values.

```bash
python create-folders/generate.py --path-images 'retinopathy-dataset/dataset/' \\\
       --path-images-idrid 'retinopathy-dataset/idrid-crop/idrid_crop/' \\\
       --path-images-messidor 'retinopathy-dataset/messidor-crop/messidor_crop/' \\\
       --path-good-quality-train 'retinopathy-dataset/train/good/' \\\
       --path-bad-quality-train 'retinopathy-dataset/train/bad/' \\\
       --path-good-quality-val 'retinopathy-dataset/val/good/' \\\
       --path-bad-quality-val 'retinopathy-dataset/val/bad/' \\\
       --path-good-quality-test 'retinopathy-dataset/test/good/' \\\
       --path-bad-quality-test 'retinopathy-dataset/test/bad/' \\\
       --csv-path 'retinopathy-dataset/result.csv'
```
It will verify there are no duplicates across different folders

```bash
python create-folder/verify_no_duplicates.py --base-path 'retinopathy-dataset'
```

# Lighning

It will train models using transfer learning, it supports inceptionV3, mobilenetV2, resnet50 and vgg19

```bash
python lightning/train.py --model 'inceptionV3' \\\
       --train-path 'retinopathy-dataset/train' \\\
       --val-path 'retinopathy-dataset/val' \\\
       --save-path 'retinopathy-dataset/board'
```