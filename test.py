import os
import csv
from pickle import TRUE

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader

test_folder = './validation'
model_file = "csv_classfier_over.pth"
USE_CUDA = 0

#dataset class
class CSVDataset(data.Dataset):
    def __init__(self, data_path, max_length=float("inf"), max_row_length = 1000):
        super(CSVDataset, self).__init__()
        self.data_path = data_path
        self.csv_files = sorted(self.make_dataset(data_path, max_length))
        self.max_row_length = max_row_length

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, index):
        with open(self.csv_files[index], newline='') as f:
            reader = csv.reader(f, delimiter=',')
            reader_int = []
            for r in reader:
                reader_int.append(list(map(int, r)))
                
            data = torch.tensor(reader_int)
            empty = torch.zeros(self.max_row_length - data.shape[0], data.shape[1])
            data = torch.cat((data,empty),dim=0).unsqueeze(0)
                
            return data, self.csv_files[index]

    def is_csv_file(self, filename):
        return filename.endswith('.csv')

    def make_dataset(self, dir, max_dataset_size=float("inf")):
        csv_files = []
        assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
            for fname in fnames:
                if self.is_csv_file(fname):
                    path = os.path.join(root, fname)
                    csv_files.append(path)
        return csv_files[:min(max_dataset_size, len(csv_files))]

#model class
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 64, 3),
                                nn.ReLU(),
                                nn.Conv2d(64, 256, 3),
                                nn.ReLU(),
                                nn.Conv2d(256, 512, 3))

        self.layer2 = nn.Sequential(nn.Linear(512,256),
                                    nn.ReLU(),
                                    nn.Linear(256,64),
                                    nn.ReLU(),
                                    nn.Linear(64, 1),
                                    nn.Sigmoid())

    def forward(self, x):
        feature = self.layer1(x).squeeze(-1)
        out = self.layer2(torch.max(feature, dim=-1)[0])
        return out

device = torch.device("cuda" if USE_CUDA else "cpu")
model = CNN().to(device)
load_dict = torch.load(model_file, map_location=device)
model.load_state_dict(load_dict)
model.eval()

test_dataset = CSVDataset(test_folder)
test_loader = DataLoader(
            test_dataset, 1, num_workers=0)


for data, file_name in test_loader:
    pred = model(data.to(device))
    if pred >= 0.5:
        print(f'prediction result:immature // file:{file_name}')
    else:
        print(f'prediction result:mature // file:{file_name}')