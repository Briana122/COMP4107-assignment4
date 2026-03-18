import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms

# PyTorch dataset for the Linnaeus 5 dataset
class Linnaeus5Dataset(torch.utils.data.Dataset):

  def __init__(self, dataset_directory):
    # dataset_directory is the full path to the directory containing the dataset
      self.image_paths = []

      # Walk through all subdirectories
      for root, dirs, files in os.walk(dataset_directory):
          for file in files:
              if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
                  self.image_paths.append(os.path.join(root, file))

      # Define preprocessing transformations
      self.transform = transforms.Compose([
          transforms.Resize((32, 32)),        # Ensure correct size
          transforms.ToTensor(),              # Convert to tensor [0,1]
          # Optional normalization (safe choice)
          # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])
    # Return nothing    

  def __len__(self):
    # num_samples is the total number of samples in the dataset
    num_samples = len(self.image_paths)
    return num_samples


  def __getitem__(self, index):
    # index is the index of the sample to be retrieved
    img_path = self.image_paths[index]
    image = Image.open(img_path).convert("RGB")
    
    # Preprocessing
    # x is one sample of data
    x = self.transform(image)

    # y is the same sample of data (autoencoder target is the same as input)
    y = x

    return x, y


# A function that creates a cnn autoencoder model for images from the Linnaeus 5 dataset
def linnaeus5_autoencoder(training_data_directory):
  # training_data_directory is the path to a directory containing the training data
  dataset = Linnaeus5Dataset(training_data_directory)

  # split into training and validation sets (80 and 20)
  train_size = int(0.8 * len(dataset))
  val_size = len(dataset) - train_size
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  # dataloaders
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

  # model is a trained cnn autoencoder model for this task
  class Autoencoder(torch.nn.Module):
    def __init__(self):
      super().__init__()

      # Encoder (Conv2D)
      self.encoder = torch.nn.Sequential(
          torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 32→16
          torch.nn.ReLU(),
          torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 16→8
          torch.nn.ReLU()
      )

      # Decoder (ConvTranspose2D)
      self.decoder = torch.nn.Sequential(
          torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # 8→16
          torch.nn.ReLU(),
          torch.nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),  # 16→32
          torch.nn.Sigmoid()  # output in [0,1]
      )

    def forward(self, x):
      x = self.encoder(x)
      x = self.decoder(x)
      return x

  model = Autoencoder()

  # Loss and optimizer
  criterion = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  # Training loop
  num_epochs = 10

  for epoch in range(num_epochs):
      model.train()
      train_loss = 0

      for x, y in train_loader:
          optimizer.zero_grad()
          outputs = model(x)
          loss = criterion(outputs, y)
          loss.backward()
          optimizer.step()

          train_loss += loss.item()

  # Compute final training performance
  model.eval()
  training_loss = 0
  with torch.no_grad():
      for x, y in train_loader:
          outputs = model(x)
          loss = criterion(outputs, y)
          training_loss += loss.item()
  # training_performance is the performance of the model on the training set
  training_performance = training_loss / len(train_loader)

  # Compute validation performance
  validation_loss = 0
  with torch.no_grad():
      for x, y in val_loader:
          outputs = model(x)
          loss = criterion(outputs, y)
          validation_loss += loss.item()

  # validation_performance is the performance of the model on the validation set
  validation_performance = validation_loss / len(val_loader)

  return model, training_performance, validation_performance
