import torch
import matplotlib.pyplot as plt
from assignment4 import Linnaeus5Dataset
from torch.utils.data import DataLoader, random_split

# Reuse your model but allow parameters to vary
def train_autoencoder(dataset_path, encoding_channels=32, num_layers=2, num_epochs=10):
    dataset = Linnaeus5Dataset(dataset_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    class Autoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()

            encoder_layers = []
            in_channels = 3

            # Build encoder dynamically
            for i in range(num_layers):
                out_channels = encoding_channels // (2 ** (num_layers - i - 1))
                encoder_layers.append(torch.nn.Conv2d(in_channels, out_channels, 3, 2, 1))
                encoder_layers.append(torch.nn.ReLU())
                in_channels = out_channels

            self.encoder = torch.nn.Sequential(*encoder_layers)

            # Build decoder dynamically (reverse)
            decoder_layers = []
            for i in range(num_layers):
                out_channels = 3 if i == num_layers - 1 else in_channels // 2
                decoder_layers.append(torch.nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1))
                if i != num_layers - 1:
                    decoder_layers.append(torch.nn.ReLU())
                in_channels = out_channels

            self.decoder = torch.nn.Sequential(*decoder_layers)

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    model = Autoencoder()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        training_performance = total_train_loss / len(train_loader)

    # Compute validation performance
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            validation_loss += loss.item()

    validation_performance = validation_loss / len(val_loader)

    return model, training_performance, validation_performance


def evaluate_on_test(model, test_dataset_path):
    test_dataset = Linnaeus5Dataset(test_dataset_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    criterion = torch.nn.MSELoss()
    model.eval()

    total_test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            total_test_loss += loss.item()

    test_performance = total_test_loss / len(test_loader)
    return test_performance


# =========================
# EXPERIMENTS
# =========================

train_dataset_path = "Linnaeus 5 32X32/train"
test_dataset_path = "Linnaeus 5 32X32/test"

best_model = None
best_params = None
best_val_loss = float("inf")
best_train_loss = None

# (a) Encoding size experiment
encoding_sizes = [8, 16, 32, 64, 128]
train_results = []
val_results = []

for size in encoding_sizes:
    model, train_loss, val_loss = train_autoencoder(
        train_dataset_path,
        encoding_channels=size
    )
    train_results.append(train_loss)
    val_results.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_train_loss = train_loss
        best_model = model
        best_params = {
            "experiment": "encoding size",
            "encoding_channels": size,
            "num_layers": 2,
            "num_epochs": 10
        }

plt.plot(encoding_sizes, train_results, marker='o', label="Train")
plt.plot(encoding_sizes, val_results, marker='o', linestyle='--', label="Validation")
plt.xlabel("Encoding Size")
plt.ylabel("Final Loss")
plt.title("Encoding Size vs Performance")
plt.legend()
plt.show()


# (b) Number of layers experiment
layers_list = [1, 2, 3, 4, 5]
train_results = []
val_results = []

for layers in layers_list:
    model, train_loss, val_loss = train_autoencoder(
        train_dataset_path,
        num_layers=layers
    )
    train_results.append(train_loss)
    val_results.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_train_loss = train_loss
        best_model = model
        best_params = {
            "experiment": "number of layers",
            "encoding_channels": 32,
            "num_layers": layers,
            "num_epochs": 10
        }

plt.plot(layers_list, train_results, marker='o', label="Train")
plt.plot(layers_list, val_results, marker='o', linestyle='--', label="Validation")
plt.xlabel("Number of Layers")
plt.ylabel("Final Loss")
plt.title("Layers vs Performance")
plt.legend()
plt.show()


# (c) Epoch experiment
epoch_values = [5, 10, 15, 20, 25]
all_train_loss = []
all_val_loss = []

for epochs in epoch_values:
    model, train_loss, val_loss = train_autoencoder(
        train_dataset_path,
        num_epochs=epochs
    )
    all_train_loss.append(train_loss)
    all_val_loss.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_train_loss = train_loss
        best_model = model
        best_params = {
            "experiment": "number of epochs",
            "encoding_channels": 32,
            "num_layers": 2,
            "num_epochs": epochs
        }

plt.plot(epoch_values, all_train_loss, marker='o', label="Train")
plt.plot(epoch_values, all_val_loss, marker='o', linestyle='--', label="Validation")
plt.xlabel("Number of Epochs")
plt.ylabel("Final Loss")
plt.title("Epochs vs Performance")
plt.legend()
plt.show()


# (d) Evaluate best model on held-out test set
test_loss = evaluate_on_test(best_model, test_dataset_path)

print("Best model parameters:", best_params)
print("Best training loss:", best_train_loss)
print("Best validation loss:", best_val_loss)
print("Test loss of best model:", test_loss)