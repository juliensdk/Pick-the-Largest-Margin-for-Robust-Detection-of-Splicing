import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import h5py
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

#device
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"The device used is {device}")

#Layers to analyze
layers = ['ConvRes', 'conv1', 'conv2', 'fc1', 'fc2', 'fc3']

# Function that computes the margin
# Function that computes the margin
def compute_layer_metrics(model, images_loader, device, layers=['ConvRes', 'conv1', 'conv2', 'fc1', 'fc2'], dist_norm=2):
    margins = {layer: [] for layer in layers}
    activations_dict = {layer: [] for layer in layers}
    model.to(device)

    with torch.no_grad():
        for images, label in images_loader:
            images, label = images.to(device), label.to(device)
            with torch.set_grad_enabled(True):
                output, activations = model(images)

            correct_class_scores = output[range(len(output)), label]
            incorrect_class_indices = 1 - label
            incorrect_class_scores = output[range(len(output)), incorrect_class_indices]
            margin_numerators = correct_class_scores - incorrect_class_scores

            for layer in layers:
                if layer not in activations:
                    print(f"Layer {layer} not found in activations.")
                    continue

                # Compute the gradients and norms for each layer
                layer_activations = activations[layer]
                activations_dict[layer].append(layer_activations.cpu())  # Store activations for total variation calculation
                layer_activations.requires_grad_(True)  # Ensure requires_grad is True

                grad_outputs = torch.zeros_like(output).to(device)  # Initialize with 0. Same size as output.
                grad_outputs[range(len(output)), label] = 1  # grad_outputs is set to 1 for the correct class
                grad_outputs[range(len(output)), incorrect_class_indices] = -1  # -1 for the incorrect class

                grads = torch.autograd.grad(outputs=output, inputs=layer_activations,
                                            grad_outputs=grad_outputs, retain_graph=True, create_graph=False, allow_unused=True)
                if grads[0] is None:
                    print(f"Warning: Gradient for layer {layer} is None.")
                    continue
                if dist_norm == 2:
                    layer_norms = torch.sqrt(torch.sum(grads[0] ** 2, dim=[i for i in range(1, grads[0].dim())]) + 1e-6)
                elif dist_norm == 1:
                    layer_norms = torch.sum(torch.abs(grads[0]), dim=[i for i in range(1, grads[0].dim())]) + 1e-6
                else:
                    raise ValueError("Unsupported norm type")

                new_margins = margin_numerators / layer_norms
                margins[layer].extend(new_margins.cpu().numpy())

    total_variation_dict = {}
    for layer, acts in activations_dict.items():
        if acts:
            all_activations = torch.cat(acts, dim=0)  # Concatenate all the activations
            response_flat = all_activations.view(all_activations.size(0), -1)  # Flatten them
            response_std = torch.std(response_flat, dim=0)  # Compute the standard deviation
            total_variation_unnormalized = torch.sqrt(torch.sum(response_std ** 2))  # Norm L2 of the standard deviation
            total_variation = total_variation_unnormalized / all_activations.size(0)  # Normalize the total variation by dividing by the number of samples
            total_variation_dict[layer] = total_variation.item()  # Store the result as a Python float

    # Normalize margins with total variation
    normalized_margins = {layer: [] for layer in layers}
    for layer in layers:
        if layer in margins and layer in total_variation_dict:
            for margin in margins[layer]:
                normalized_margins[layer].append(margin / total_variation_dict[layer])
                # print(f"margin before normalization for {layer} : {margin} ")
            #    print(f"Total variation for {layer} is {total_variation_dict[layer]}")

    return normalized_margins

# Define the CNN architecture
class ConvRes(nn.Module):  # Creation of the custom first convolutional layer
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvRes, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))  # 4D tensor with random weights
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -0.1, 0.1)  # initize all parameters with values between -0.1 , 0.1
        self.apply_constraints()

    def apply_constraints(self):
        with torch.no_grad():  # Ensures that operations inside this block don't track gradients
            self.weight[:, :, self.kernel_size // 2, self.kernel_size // 2] = -1
            epsilon = 1e-8
            for i in range(self.out_channels):  # iteration over all filters
                for j in range(self.in_channels):  # iteration over every input feature layer
                    central_weight = self.weight[i, j, self.kernel_size // 2, self.kernel_size // 2]
                    sum_weights = self.weight[i, j].sum() - central_weight  # calculates the sum of all the weights in the current filter for the current input channel and exclude the central pixel of the total sum of weights
                    self.weight[i, j] /= (sum_weights + epsilon)  # normalization of the weights of the current filter for the current input channel, by diving them by the previous sum so that the sum =1

    def forward(self, x):
        self.apply_constraints()  # We apply it at every forward pass, in addition to the initialization
        return F.conv2d(x, self.weight, stride=(self.stride, self.stride), padding=(self.padding, self.padding))

def accuracy(output, target):
    preds = torch.argmax(output, dim=1)
    correct = preds.eq(target).sum().item()
    acc = correct / target.size(0)
    return torch.tensor(acc)

class Net(pl.LightningModule):
    def __init__(self, dropout_rate, normalization, pooling):
        super(Net, self).__init__()

        self.dropout_rate = dropout_rate
        self.normalization = normalization
        self.pooling = pooling

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
        activations = {}
        x = self.ConvRes(x)
        activations['ConvRes'] = x
        x = self.conv1(x)
        activations['conv1'] = x
        x = self.pool(x)

        if self.normalization != 'local_response':
            x = self.norm1(x)
            x = F.relu(x)
        else:
            x = self.norm1(x)

        x = self.conv2(x)
        activations['conv2'] = x
        x = self.pool(x)

        if self.normalization != 'local_response':
            x = self.norm2(x)
            x = F.relu(x)
        else:
            x = self.norm2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        activations['fc1'] = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate)
        x = self.fc2(x)
        activations['fc2'] = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate)
        x = self.out(x)
        activations['fc3'] = x
        return x, activations

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs, _ = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        acc = accuracy(outputs, labels)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)  # Log test loss
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)  # Log test accuracy
        return {'test_loss': loss, 'test_acc': acc}

# Images Loader
class CustomH5imagesset(Dataset):
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

#Function that extracts the number of the model (ex : model_1.ckpt)
def extract_nb(weight_file):
    match = re.search(r'_(\d+)\.ckpt', weight_file)
    if match:
        nb = int(match.group(1))
    else :
        print("No match found")
    return nb

# h5 file
evaluation_h5_path_1 = "h5_files_evaluation/evaluation_Sans_Compression.h5"

# The csv with the informations about the training of the models
csv_path = "training_results/training_results.csv"
# Dir to store the margins
csv_dir = "margin_results_2"
# Dir where are located the model parameters
parameter_dir = "parametres_2"

signature_abs = {}
signature_positive = {}

#Iterates over all models
for weight_file in os.listdir(parameter_dir):
    if weight_file.startswith("."):
        continue

    weight_path = os.path.join(parameter_dir, weight_file)
    print(f"Processing {weight_path}")

    #Get the number of the model
    nb = extract_nb(weight_file)
    if nb is None:
        continue

    # Extract its training hyperparameters from the results csv
    print(csv_path)
    df = pd.read_csv(csv_path)
    #print(df.head())

    # Ensure the columns are present in the DataFrame
    required_columns = ['Dropout Rate', 'Pooling', 'Normalization', 'Batch Size']
    for col in required_columns:
        if col not in df.columns:
            print(f"Column '{col}' not found in the CSV file.")
            continue

    # Access the values safely
    Dropout_Rate = df.iloc[nb - 1]['Dropout Rate']
    Pooling = df.iloc[nb - 1]['Pooling']
    Normalization = df.iloc[nb - 1]['Normalization']
    Batch = df.iloc[nb - 1]['Batch Size']

    print(f"Dropout_Rate : {Dropout_Rate} - Pooling : {Pooling} - Normalization : {Normalization} - Batch : {Batch}")

    # Verify if the checkpoint file exists and is valid
    if not os.path.exists(weight_path):
        print(f"Checkpoint file {weight_path} does not exist.")
        continue
    if os.path.getsize(weight_path) == 0:
        print(f"Checkpoint file {weight_path} is empty.")
        continue

    try:
        # Load the weights of the model
        checkpoint = torch.load(weight_path, map_location=device)
    except Exception as e:
        print(f"Failed to load checkpoint {weight_path}: {e}")
        continue

    state_dict = checkpoint['state_dict']

    # Instantiate the model
    model = Net(Dropout_Rate, Normalization, Pooling)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    #Evaluation sur la distribution d'entrainement
    evaluationset = CustomH5imagesset(evaluation_h5_path_1, 'evaluation_images', 'evaluation_labels', transform=transform)
    evaluationloader1 = DataLoader(evaluationset, batch_size=128, shuffle=False, num_workers=0)
    
    #Compute the margins for all layer
    normalized_margins = compute_layer_metrics(model, evaluationloader1, device, layers, 2)

    # Save results as csv
    df2 = pd.DataFrame(normalized_margins)
    csv_model_name = f'{os.path.basename(weight_file).split(".")[0]}.csv'
    new_csv_path = os.path.join(csv_dir, csv_model_name)
    df2.to_csv(new_csv_path, index=False)

    print(f"Finished with model : {weight_file}")
