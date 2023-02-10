import torch

from images_dataset_predict import Dataset
from torch.utils.data import DataLoader
import albumentations as A
import albumentations.pytorch.transforms as AT
import argparse
import importlib
import os
import pandas as pd

def compactDf(df):
    goodImgDf = df[['good_images_names',
                    'good_images_probability_left',
                    'good_images_probability_right',
                    'good_images_predicts']]

    blurDf = df[['blur_names',
                 'blur_probability_left',
                 'blur_probability_right',
                 'blur_predicts']]

    gaussDf = df[['gauss_noise_names',
                  'gauss_noise_probability_left',
                  'gauss_noise_probability_right',
                  'gauss_noise_predicts']]

    randomFogDf = df[['random_fog_names',
                      'random_fog_probability_left',
                      'random_fog_probability_right',
                      'random_fog_predicts']]

    goodImgDf['good_images_names'] = goodImgDf['good_images_names'].str.replace('.jpg', '.png')

    goodImgDf.dropna(how='all', inplace=True)
    blurDf.dropna(how='all', inplace=True)
    gaussDf.dropna(how='all', inplace=True)
    randomFogDf.dropna(how='all', inplace=True)

    return goodImgDf\
        .merge(blurDf, left_on='good_images_names', right_on='blur_names', how='inner')\
        .merge(gaussDf, left_on='good_images_names', right_on='gauss_noise_names', how='inner')\
        .merge(randomFogDf, left_on='good_images_names', right_on='random_fog_names', how='inner')




def main():
    inceptionDf = pd.read_csv(os.path.join(os.getcwd(), args.inception_predicts))
    mobileNetDf = pd.read_csv(os.path.join(os.getcwd(), args.mobilenet_predicts))
    resnetDf = pd.read_csv(os.path.join(os.getcwd(), args.resnet_predicts))
    vggDf = pd.read_csv(os.path.join(os.getcwd(), args.vgg_predicts))

    merged = compactDf(inceptionDf)

    merged.to_csv(os.path.join(os.getcwd(), args.output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inception-predicts', help='Path to inception csv')
    parser.add_argument('--mobilenet-predicts', help='Path to mobilenet csv')
    parser.add_argument('--resnet-predicts', help='Path to resnet csv')
    parser.add_argument('--vgg-predicts', help='Path to vgg csv')
    parser.add_argument('--output', help='Path to output csv')

    args = parser.parse_args()
    main()
