import albumentations as A
import cv2
import random
import os
import re

read_path = "/Users/pbermejo/Documents/Master/retinopathy-dataset/test/good/"
write_good_sample_path = "/Users/pbermejo/Documents/Master/retinopathy-dataset/test/good_sample/"
write_bad_blur_sample_path = "/Users/pbermejo/Documents/Master/retinopathy-dataset/test/bad_blur_sample/"
write_bad_gauss_noise_sample_path = "/Users/pbermejo/Documents/Master/retinopathy-dataset/test/bad_gauss_noise_sample/"
write_bad_random_fog_sample_path = "/Users/pbermejo/Documents/Master/retinopathy-dataset/test/bad_random_fog_sample/"

listGoodImages = [
            name for name in os.listdir(read_path)
            if not re.match(r'[\w,\d]+\.[json]{4}', name)
        ]

sampleGoodImages = random.sample(listGoodImages, 100)

# print(sampleGoodImages)

transform_blur = A.Compose([
    A.Blur(always_apply=True, p=1, blur_limit=(13, 15))
])

transform_gauss_noise = A.Compose([
    A.GaussNoise(always_apply=True, p=1, var_limit=(10.0, 100.0))
])

transform_random_fog = A.Compose([
    A.RandomFog(always_apply=True, p=1, fog_coef_lower=0.1, fog_coef_upper=0.2, alpha_coef=0.40)
])

for imageName in sampleGoodImages:
    image = cv2.imread(os.path.join(read_path, imageName))
    cv2.imwrite(os.path.join(write_good_sample_path, imageName), image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformedBlurImage = transform_blur(image=image)["image"]
    transformedGaussNoiseImage = transform_gauss_noise(image=image)["image"]
    transformedRandomFogImage = transform_random_fog(image=image)["image"]

    cv2.imwrite(os.path.join(write_bad_blur_sample_path, imageName), transformedBlurImage)
    cv2.imwrite(os.path.join(write_bad_gauss_noise_sample_path, imageName), transformedGaussNoiseImage)
    cv2.imwrite(os.path.join(write_bad_random_fog_sample_path, imageName), transformedRandomFogImage)




