import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pytorch_lightning import Trainer, seed_everything
import torchvision.transforms as transforms
import os
import h5py
import yaml
import pandas as pd


model_name_csv = {}

parametres_dir_path = "parametres"
csv_path = "training_results/training_results.csv"
config_dir_path = "config"

multiprocessing.set_start_method('fork', force=True)
torch.use_deterministic_algorithms(True, warn_only=True)

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"The device used is {device}")

classes = ('false', 'true')

# Import the h5 file
class CustomH5Dataset(Dataset):
    def __init__(self, h5_file, images_name, labels_name, transform=None):
        self.h5_file = h5_file
        self.transform = transform
        self.images = images_name
        self.labels = labels_name
        with h5py.File(self.h5_file, 'r') as file:
            self.length = len(file[self.images])

    def __getitem__(self, index):
        with h5py.File(self.h5_file, 'r') as file:
            image = file[self.images][index]
            label = int(file[self.labels][index])
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.length

transform = transforms.Compose(
    [transforms.ToTensor(),  # tensorize the image
     transforms.Normalize((0.5,), (0.5,))])  # normalize it btw -1 to 0, instead of 0 to 255

# Define the constrained convolutional layer
class ConvRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvRes, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -0.1, 0.1)
        self.apply_constraints()

    def apply_constraints(self):
        with torch.no_grad():
            self.weight[:, :, self.kernel_size // 2, self.kernel_size // 2] = -1
            epsilon = 1e-8
            for i in range(self.out_channels):
                for j in range(self.in_channels):
                    central_weight = self.weight[i, j, self.kernel_size // 2, self.kernel_size // 2]
                    sum_weights = self.weight[i, j].sum() - central_weight
                    self.weight[i, j] /= (sum_weights + epsilon)

    def forward(self, x):
        self.apply_constraints()
        return F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

 # Define the function that returns the accuracy
def accuracy(output, target):
    preds = torch.argmax(output, dim=1)
    correct = preds.eq(target).sum().item()
    acc = correct / target.size(0)
    return torch.tensor(acc)

class Net(pl.LightningModule):
    def __init__(self, dropout_rate, normalization, pooling):
        super(Net, self).__init__()

        self.normalization = normalization
        self.pooling = pooling
        self.dropout_rate = dropout_rate

        self.ConvRes = ConvRes(3, 12, 5)
        self.conv1 = nn.Conv2d(12, 64, 7, 2, 3)

        if self.pooling == 'max_pooling':
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif self.pooling == 'average_pooling':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 48, 5, 1, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(12288, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.out = nn.Linear(4096, 2)

        if self.normalization == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
            self.norm2 = nn.BatchNorm2d(48)
        elif self.normalization == 'layer':
            self.norm1 = nn.LayerNorm([64, 31, 31])
            self.norm2 = nn.LayerNorm([48, 16, 16])
        elif self.normalization == 'group':
            self.norm1 = nn.GroupNorm(8, 64)
            self.norm2 = nn.GroupNorm(8, 48)
        elif self.normalization == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
            self.norm2 = nn.InstanceNorm2d(48)
        elif self.normalization == 'local_response':
            self.norm1 = nn.LocalResponseNorm(size=5)
            self.norm2 = nn.LocalResponseNorm(size=5)

    def forward(self, x):
        x = self.ConvRes(x)
        x = self.conv1(x)
        x = self.pool(x)

        if self.normalization != 'local_response':
            x = self.norm1(x)
            x = F.relu(x)
        else:
            x = self.norm1(x)

        x = self.conv2(x)
        x = self.pool(x)

        if self.normalization != 'local_response':
            x = self.norm2(x)
            x = F.relu(x)
        else:
            x = self.norm2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate)
        x = self.out(x)
        return x

    def training_step(self, batch):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        acc = accuracy(outputs, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        if not hasattr(self, 'validation_losses'):
            self.validation_losses = []
        self.validation_losses.append(loss.detach())
        return {"loss": loss, "accuracy": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

for i, config_file_name in enumerate(os.listdir(config_dir_path)):

    if not config_file_name.endswith(".yaml"):
        print(f"Skipping non-yaml file: {config_file_name}")
        continue

    # Load the config file and extract the values
    def load_config(config_path):
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config

    config_file_path = os.path.join(config_dir_path, config_file_name)
    config = load_config(config_file_path)

    # Affect values
    seed_everything(config['seed'], workers=True)
    batch_size = config['batch_size']
    normalization = config['normalization']
    pooling = config['pooling']
    dropout_rate = config['dropout']


    # Load the two datasets for training and early stopping
    Training_path = 'h5_files/train_Sans_Compression.h5'
    train_dataset = CustomH5Dataset(Training_path, 'train_images', 'train_labels', transform=transform)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    Operational_path = 'h5_files/operational_Sans_Compression.h5'
    operationalset = CustomH5Dataset(Operational_path, 'operational_images', 'operational_labels', transform=transform)
    operationaloader = torch.utils.data.DataLoader(operationalset, batch_size=batch_size, shuffle=False, num_workers=2)


    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=15,
        verbose=True,
        mode='min')

    model_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=parametres_dir_path,
        filename=f'model_{i + 1}',
        save_top_k=1,
        mode='min',
        verbose=True,
    )

    net = Net(dropout_rate, normalization, pooling)

    trainer = pl.Trainer(
        callbacks=[early_stop_callback, model_checkpoint_callback],
        max_epochs=config['epochs'],
    )

    trainer.fit(model=net, train_dataloaders=trainloader, val_dataloaders=operationaloader)

    #Modif les valeurs du csv
    model_name_csv[f'Model {i+1}'] = [config['seed'], batch_size, dropout_rate, config['epochs'], pooling, normalization, f'{trainer.current_epoch:02d}',f'{trainer.callback_metrics["val_loss"].item():.3f}', f'{trainer.callback_metrics["val_acc"].item():.3f}']
    print(f"finished with model {i+1} - {model_name_csv[f'Model {i+1}']}")

#S'assurer ici que tout va bien.
df = pd.DataFrame.from_dict(model_name_csv, orient='index', columns=['Seed', 'Batch Size', 'Dropout Rate', 'Epochs', 'Pooling', 'Normalization', 'Last Epoch', 'Val Loss', 'Val Acc'])
df.to_csv(csv_path, index=True)

print(f"Finished the {i} training - results saved at {csv_path}")