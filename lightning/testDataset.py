from images_dataset import Dataset
import albumentations
import albumentations.pytorch.transforms as AT

path = "/Users/pedro_bermejo/Documents/Master/retinopathy-dataset/val"

transform = albumentations.Compose([
            albumentations.Resize(width=256, height=256),
            albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            AT.ToTensorV2()
        ])

dataset = Dataset(path, transform)

print(len(dataset))
print(dataset.__getitem__(0))
