from os import makedirs
from os.path import join
from os.path import exists
import albumentations as aug

def create_directory(directory_name):
        try:
            full_path = join(directory_name)
            if not exists(full_path):
                makedirs(full_path)
        except IOError:
            raise

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    transform = [
        aug.Lambda(image=preprocessing_fn),
        aug.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return aug.Compose(transform)