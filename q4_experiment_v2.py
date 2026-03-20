import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms


class Linnaeus5DatasetQ4(Dataset):
    def __init__(self, dataset_directory, preprocessing="zero_one"):
        self.image_paths = []

        for root, dirs, files in os.walk(dataset_directory):
            for file in files:
                if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
                    self.image_paths.append(os.path.join(root, file))

        # Q4(a) preprocessing options:
        # 1) binning: reduce image to binary-like values after tensor conversion
        # 2) zero_one: pixel values in [0, 1]
        # 3) mean_std: mean-zero, std-1 style scaling using mean=0.5, std=0.5
        if preprocessing == "binning":
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x > 0.5).float())
            ])
        elif preprocessing == "zero_one":
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])
        elif preprocessing == "mean_std":
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        else:
            raise ValueError(f"Unknown preprocessing option: {preprocessing}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        x = self.transform(image)
        y = x
        return x, y


class Autoencoder(torch.nn.Module):
    def __init__(self, batchnorm_momentum=0.0, dropout_rate=0.0, output_activation="sigmoid"):
        super().__init__()

        use_batchnorm = batchnorm_momentum is not None

        encoder_layers = [torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)]
        if use_batchnorm:
            encoder_layers.append(torch.nn.BatchNorm2d(16, momentum=batchnorm_momentum))
        encoder_layers.append(torch.nn.ReLU())
        if dropout_rate > 0:
            encoder_layers.append(torch.nn.Dropout2d(dropout_rate))

        encoder_layers.append(torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1))
        if use_batchnorm:
            encoder_layers.append(torch.nn.BatchNorm2d(32, momentum=batchnorm_momentum))
        encoder_layers.append(torch.nn.ReLU())
        if dropout_rate > 0:
            encoder_layers.append(torch.nn.Dropout2d(dropout_rate))

        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = [torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)]
        if use_batchnorm:
            decoder_layers.append(torch.nn.BatchNorm2d(16, momentum=batchnorm_momentum))
        decoder_layers.append(torch.nn.ReLU())
        if dropout_rate > 0:
            decoder_layers.append(torch.nn.Dropout2d(dropout_rate))

        decoder_layers.append(torch.nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1))

        if output_activation == "sigmoid":
            decoder_layers.append(torch.nn.Sigmoid())
        elif output_activation == "tanh":
            decoder_layers.append(torch.nn.Tanh())
        else:
            raise ValueError(f"Unknown output activation: {output_activation}")

        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_output_activation(preprocessing):
    if preprocessing in ["binning", "zero_one"]:
        return "sigmoid"
    elif preprocessing == "mean_std":
        return "tanh"
    else:
        raise ValueError(f"Unknown preprocessing option: {preprocessing}")


def train_autoencoder(
    dataset_path,
    preprocessing="zero_one",
    batchnorm_momentum=None,
    regularization=None,
    reg_lambda=0.0,
    dropout_rate=0.0,
    learning_rate=0.001,
    num_epochs=10,
    seed=42,
):
    torch.manual_seed(seed)

    dataset = Linnaeus5DatasetQ4(dataset_path, preprocessing=preprocessing)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Autoencoder(
        batchnorm_momentum=batchnorm_momentum,
        dropout_rate=dropout_rate,
        output_activation=get_output_activation(preprocessing)
    )

    criterion = torch.nn.MSELoss()
    weight_decay = reg_lambda if regularization == "l2" else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)

            if regularization == "l1":
                l1_penalty = 0.0
                for param in model.parameters():
                    l1_penalty += torch.sum(torch.abs(param))
                loss = loss + reg_lambda * l1_penalty

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        training_performance = total_train_loss / len(train_loader)

    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            validation_loss += loss.item()

    validation_performance = validation_loss / len(val_loader)
    return model, training_performance, validation_performance


def evaluate_on_test(model, test_dataset_path, preprocessing="zero_one"):
    test_dataset = Linnaeus5DatasetQ4(test_dataset_path, preprocessing=preprocessing)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    criterion = torch.nn.MSELoss()
    model.eval()

    total_test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            total_test_loss += loss.item()

    return total_test_loss / len(test_loader)


def run_experiment(x_values, train_values, val_values, xlabel, title):
    plt.figure()
    plt.plot(x_values, train_values, marker="o", label="Train")
    plt.plot(x_values, val_values, marker="o", linestyle="--", label="Validation")
    plt.xlabel(xlabel)
    plt.ylabel("Final Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


train_dataset_path = "Linnaeus 5 32X32/train"
test_dataset_path = "Linnaeus 5 32X32/test"

best_model = None
best_params = None
best_val_loss = float("inf")
best_train_loss = None
best_preprocessing = "zero_one"


# =========================
# (a) Preprocessing experiment
# =========================
preprocessing_options = ["binning", "zero_one", "mean_std"]
train_results = []
val_results = []

for option in preprocessing_options:
    model, train_loss, val_loss = train_autoencoder(
        train_dataset_path,
        preprocessing=option
    )
    train_results.append(train_loss)
    val_results.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_train_loss = train_loss
        best_model = model
        best_preprocessing = option
        best_params = {
            "experiment": "preprocessing",
            "preprocessing": option,
            "batchnorm_momentum": None,
            "regularization": None,
            "reg_lambda": 0.0,
            "dropout_rate": 0.0,
            "learning_rate": 0.001,
            "num_epochs": 10,
        }

run_experiment(
    preprocessing_options,
    train_results,
    val_results,
    "Preprocessing",
    "Preprocessing vs Performance"
)


# =========================
# (b) Batch normalization experiment
# testing different momentum values
# =========================
batchnorm_values = [None, 0.05, 0.1, 0.2, 0.5]
batchnorm_labels = ["none", "0.05", "0.1", "0.2", "0.5"]
train_results = []
val_results = []

for momentum in batchnorm_values:
    model, train_loss, val_loss = train_autoencoder(
        train_dataset_path,
        preprocessing="zero_one",
        batchnorm_momentum=momentum
    )
    train_results.append(train_loss)
    val_results.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_train_loss = train_loss
        best_model = model
        best_preprocessing = "zero_one"
        best_params = {
            "experiment": "batch normalization",
            "preprocessing": "zero_one",
            "batchnorm_momentum": momentum,
            "regularization": None,
            "reg_lambda": 0.0,
            "dropout_rate": 0.0,
            "learning_rate": 0.001,
            "num_epochs": 10,
        }

run_experiment(
    batchnorm_labels,
    train_results,
    val_results,
    "BatchNorm Momentum",
    "Batch Normalization vs Performance"
)


# =========================
# (c) Regularization experiment
# L1 only or L2 only
# =========================
regularization_configs = [
    ("l1", 1e-6),
    ("l1", 1e-5),
    ("l1", 1e-4),
    ("l2", 1e-6),
    ("l2", 1e-5),
    ("l2", 1e-4),
]
regularization_labels = ["L1 1e-6", "L1 1e-5", "L1 1e-4", "L2 1e-6", "L2 1e-5", "L2 1e-4"]
train_results = []
val_results = []

for reg_type, reg_lambda in regularization_configs:
    model, train_loss, val_loss = train_autoencoder(
        train_dataset_path,
        preprocessing="zero_one",
        regularization=reg_type,
        reg_lambda=reg_lambda
    )
    train_results.append(train_loss)
    val_results.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_train_loss = train_loss
        best_model = model
        best_preprocessing = "zero_one"
        best_params = {
            "experiment": "regularization",
            "preprocessing": "zero_one",
            "batchnorm_momentum": None,
            "regularization": reg_type,
            "reg_lambda": reg_lambda,
            "dropout_rate": 0.0,
            "learning_rate": 0.001,
            "num_epochs": 10,
        }

run_experiment(
    regularization_labels,
    train_results,
    val_results,
    "Regularization",
    "L1/L2 Regularization vs Performance"
)


# =========================
# (d) Dropout experiment
# =========================
dropout_values = [0.0, 0.1, 0.2, 0.3, 0.5]
train_results = []
val_results = []

for dropout in dropout_values:
    model, train_loss, val_loss = train_autoencoder(
        train_dataset_path,
        preprocessing="zero_one",
        dropout_rate=dropout
    )
    train_results.append(train_loss)
    val_results.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_train_loss = train_loss
        best_model = model
        best_preprocessing = "zero_one"
        best_params = {
            "experiment": "dropout",
            "preprocessing": "zero_one",
            "batchnorm_momentum": None,
            "regularization": None,
            "reg_lambda": 0.0,
            "dropout_rate": dropout,
            "learning_rate": 0.001,
            "num_epochs": 10,
        }

run_experiment(
    dropout_values,
    train_results,
    val_results,
    "Dropout Rate",
    "Dropout vs Performance"
)


# =========================
# (e) Learning rate experiment
# =========================
learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
train_results = []
val_results = []

for lr in learning_rates:
    model, train_loss, val_loss = train_autoencoder(
        train_dataset_path,
        preprocessing="zero_one",
        learning_rate=lr
    )
    train_results.append(train_loss)
    val_results.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_train_loss = train_loss
        best_model = model
        best_preprocessing = "zero_one"
        best_params = {
            "experiment": "learning rate",
            "preprocessing": "zero_one",
            "batchnorm_momentum": None,
            "regularization": None,
            "reg_lambda": 0.0,
            "dropout_rate": 0.0,
            "learning_rate": lr,
            "num_epochs": 10,
        }

run_experiment(
    learning_rates,
    train_results,
    val_results,
    "Learning Rate",
    "Learning Rate vs Performance"
)


# =========================
# (f) Evaluate best model on held-out test set
# =========================
test_loss = evaluate_on_test(best_model, test_dataset_path, preprocessing=best_preprocessing)

print("Best model parameters:", best_params)
print("Best training loss:", best_train_loss)
print("Best validation loss:", best_val_loss)
print("Test loss of best model:", test_loss)
