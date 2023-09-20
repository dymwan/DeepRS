# Dataset tutorial
In Pytorch, torch.utils.data.Datasets is responsible for dataset construction,
and torch.utils.data.DataLoader is respondible for data loading and transfer to 
model.

## customize dataset
override methods __getitem__ and __len__ to build a custom dataset 