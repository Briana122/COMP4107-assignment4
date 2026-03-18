# Name this file assignment4.py when you submit
import torch

# PyTorch dataset for the Linnaeus 5 dataset
class Linnaeus5Dataset(torch.utils.data.Dataset):

  def __init__(self, dataset_directory):
    # dataset_directory is the full path to the directory containing the dataset

    # Return nothing    

  def __len__(self):
    # num_samples is the total number of samples in the dataset
    return num_samples


  def __getitem__(self, index):
    # index is the index of the sample to be retrieved
    
    # x is one sample of data
    # y is the same sample of data
    return x, y


# A function that creates a cnn autoencoder model for images from the Linnaeus 5 dataset
def linnaeus5_autoencoder(training_data_directory):
  # training_data_directory is the path to a directory containing the training data

  # model is a trained cnn autoencoder model for this task
  # training_performance is the performance of the model on the training set
  # validation_performance is the performance of the model on the validation set
  return model, training_performance, validation_performance
