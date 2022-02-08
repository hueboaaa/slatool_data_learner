import pdb
import os
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#hyper parameter
epoch = 100
batch_size = 2
lr = 1e-4
num_workers = 0

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
        if "immature" in self.csv_files[index]:
            target = torch.ones(1)
        else:
            target = torch.zeros(1)

        with open(self.csv_files[index], newline='') as f:
            reader = csv.reader(f, delimiter=',')
            reader_int = []
            for r in reader:
                reader_int.append(list(map(int, r)))
                
            data = torch.tensor(reader_int)
            empty = torch.zeros(self.max_row_length - data.shape[0], data.shape[1])
            data = torch.cat((data,empty),dim=0).unsqueeze(0)
                
            return data, target

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

#data loader
train_dataset = CSVDataset('./train')
train_loader = DataLoader(
            train_dataset, batch_size, True,
            num_workers=num_workers, pin_memory=True, drop_last=True)
val_dataset = CSVDataset('./validation')
val_loader = DataLoader(
            val_dataset, batch_size, True,
            num_workers=num_workers, pin_memory=True, drop_last=True)

#model
model = CNN() #.cuda()

#loss & optimizer
loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr)

graph_train_loss    = []
graph_train_acc     = []
graph_val_loss      = []
graph_val_acc       = []

for ep in range(epoch):
    # train step
    model.train()
    train_losses = []
    train_acc = [] 
    for data, target in train_loader:
        pred = model(data) #.cuda())
        out = loss(pred, target) #.cuda())
        train_losses.append(out)

        optimizer.zero_grad()
        out.backward()
        optimizer.step()

        for p, t in zip(pred, target):
            if p >= 0.5:
                if t == 1.0:
                    train_acc.append(1)
                else:
                    train_acc.append(0)
            else:
                if t == 0.0:
                    train_acc.append(1)
                else:
                    train_acc.append(0)

    
    # validation step
    with torch.no_grad():
        model.eval()
        val_losses = []
        val_acc = []
        for data, target in val_loader:
            pred = model(data) #.cuda())
            out = loss(pred, target) #.cuda())
            val_losses.append(out)

            for p, t in zip(pred, target):
                if p >= 0.5:
                    if t == 1.0:
                        val_acc.append(1)
                    else:
                        val_acc.append(0)
                else:
                    if t == 0.0:
                        val_acc.append(1)
                    else:
                        val_acc.append(0)

    train_total_loss = sum(train_losses) / len(train_losses)
    train_total_acc = sum(train_acc) / len(train_acc)
    val_total_loss = sum(val_losses) / len(val_losses)
    val_total_acc = sum(val_acc) / len(val_acc)

    graph_train_loss.append(train_total_loss.detach().cpu().numpy())
    graph_train_acc.append(train_total_acc)
    graph_val_loss.append(val_total_loss.detach().cpu().numpy())
    graph_val_acc.append(val_total_acc)

    print(f"epoch:{ep} >> train loss:{train_total_loss} train acc:{train_total_acc} / val loss:{val_total_loss} val acc:{val_total_acc} ")    



# summarize history for accuracy
plt.plot(graph_train_acc)
plt.plot(graph_val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
# plt.show()
plt.savefig('model_acc.png', dpi=300)
plt.cla() 

# summarize history for loss
plt.plot(graph_train_loss)
plt.plot(graph_val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
# plt.show()
plt.savefig('model_loss.png', dpi=300)

save_model_name = "csv_classfier.pth"
torch.save(model.state_dict(), save_model_name)


#test input
# data = torch.rand(2,1,600,7).cuda()