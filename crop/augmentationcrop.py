import albumentations as aug


def get_training_augmentation():
    train_transform = [
        aug.HorizontalFlip(p=0.5),
        aug.OneOf(
            [
                aug.CLAHE(),
                aug.GaussNoise(),
            ],
            p=0.25,
        ),
        aug.OneOf(
            [
                aug.Sharpen(),
                aug.Blur(blur_limit=3),
                aug.MotionBlur(blur_limit=3),
            ],
            p=0.25,
        ),
        aug.RandomBrightnessContrast(p=0.25),
    ]
    return aug.Compose(train_transform)


def get_validation_augmentation():
    train_transform = [
        aug.Resize(height=512, width=512)
    ]
    return aug.Compose(train_transform)


def get_test_augmentation():
    return get_training_augmentation()
