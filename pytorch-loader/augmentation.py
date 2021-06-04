import albumentations as albu

from albumentations.pytorch.transforms import ToTensorV2


def get_train_transform():
    transforms = albu.Compose([
        albu.Resize(width=512, height=512),
        albu.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255),
        ToTensorV2()
    ])
    return transforms
