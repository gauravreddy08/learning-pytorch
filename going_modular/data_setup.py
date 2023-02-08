"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count()

def create_dataset(
            train_dir: str, test_dir: str,
            transform: transforms.Compose, 
            batch_size: int = 32,
            num_workers: int = NUM_WORKERS
          ):
  """
  Creates torch dataloaders from the given train and test directories
  for image classification tasks.

  Args: 
    train_dir: Path of training directory.
    test_dir: Path of testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch (default=32).
    num_works: An integer for number of workers per DataLoader.
  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
  """
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  class_names = train_data.classes

  train_dataloader = torch.utils.data.DataLoader(
      train_data, batch_size=batch_size,
      shuffle=True, num_workers=num_workers,
      pin_memory=True
  )
  test_dataloader = torch.utils.data.DataLoader(
      test_data, batch_size=batch_size,
      shuffle=False, num_workers=num_workers,
      pin_memory=True
  )

  return train_dataloader, test_dataloader, class_names
