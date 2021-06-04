from dataset import MyDataset
from torch.utils.data import DataLoader
from augmentation import get_train_transform


DATA_PATH = '/home/abraham/Desktop/dataset/data'

dataset = MyDataset(image_path=DATA_PATH, tranform=get_train_transform())
loader = DataLoader(dataset=dataset, shuffle=True, batch_size=4)

for data in loader:
    inputs, labels = data
    print(inputs.shape, labels)
