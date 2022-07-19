import albumentations as A
import argparse
import cv2
import random
import os
import re


def main():
    # First clear folders
    folders_to_clear = [args.good_sample_path,
                        args.bad_blur_sample_path,
                        args.bad_gauss_noise_sample_path,
                        args.bad_random_fog_sample_path]

    for folder_path in folders_to_clear:
        images = [name for name in os.listdir(os.path.join(os.getcwd(), folder_path))]
        for image_path in images:
            os.remove(os.path.join(os.getcwd(), folder_path, image_path))


    # List images in good folder and take a sample
    listGoodImages = [
                name for name in os.listdir(os.path.join(os.getcwd(), args.good_path))
                if not re.match(r'[\w,\d]+\.[json]{4}', name)
            ]

    sampleGoodImages = random.sample(listGoodImages, 100)

    # print(sampleGoodImages)

    transform_blur = A.Compose([
        A.Blur(always_apply=True, p=1, blur_limit=(17, 18))
    ])

    transform_gauss_noise = A.Compose([
        A.GaussNoise(always_apply=True, p=1, var_limit=(140.0, 150.0))
    ])

    transform_random_fog = A.Compose([
        A.RandomFog(always_apply=True, p=1, fog_coef_lower=0.1, fog_coef_upper=0.2, alpha_coef=0.40)
    ])

    # Apply albumentations and save
    for imageName in sampleGoodImages:
        image = cv2.imread(os.path.join(os.getcwd(), args.good_path, imageName))
        cv2.imwrite(os.path.join(os.getcwd(), args.good_sample_path, imageName), image)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformedBlurImage = transform_blur(image=image)["image"]
        transformedGaussNoiseImage = transform_gauss_noise(image=image)["image"]
        transformedRandomFogImage = transform_random_fog(image=image)["image"]

        cv2.imwrite(os.path.join(os.getcwd(), args.bad_blur_sample_path, imageName), transformedBlurImage)
        cv2.imwrite(os.path.join(os.getcwd(), args.bad_gauss_noise_sample_path, imageName), transformedGaussNoiseImage)
        cv2.imwrite(os.path.join(os.getcwd(), args.bad_random_fog_sample_path, imageName), transformedRandomFogImage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--good-path', help='foo help')
    parser.add_argument('--good-sample-path', help='foo help')
    parser.add_argument('--bad-blur-sample-path', help='foo help')
    parser.add_argument('--bad-gauss-noise-sample-path', help='foo help')
    parser.add_argument('--bad-random-fog-sample-path', help='foo help')
    args = parser.parse_args()
    main()
