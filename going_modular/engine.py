import torch 
from torch import nn
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score

def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
               device: torch.device='cuda' if torch.cuda.is_available() else 'cpu'):
  """
  Training Torch model and updates weights per batch.
  
  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    Tuple(train_loss, train_accuracy)
  """

  train_loss = 0
  train_acc = 0
  
  model.train()
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    preds = model(X)
    loss = loss_fn(preds, y)
    
    train_loss+=loss
    train_acc += accuracy_score(y.cpu(), torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  return train_loss/len(dataloader), train_acc/len(dataloader)

def test_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              device: torch.device='cuda' if torch.cuda.is_available() else 'cpu'):
  """
  Runs Torch model in evaluation model.
  
  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    Tuple(test_loss, test_accuracy)
  """
  test_loss = 0
  test_acc = 0

  from sklearn.metrics import accuracy_score
  
  model.eval()
  with torch.inference_mode():
    for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)
      preds = model(X)
      test_loss += loss_fn(preds, y)
      test_acc += accuracy_score(y.cpu(), torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu())
    return test_loss/len(dataloader), test_acc/len(dataloader)

def train(model: nn.Module, epochs: int,
          train_data: torch.utils.data.DataLoader,
          loss: nn.Module, optimizer: torch.optim.Optimizer,
          test_data: torch.utils.data.DataLoader, 
          device: torch.device='cuda' if torch.cuda.is_available() else 'cpu'):
  
  """
  Trains and tests a PyTorch model.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {epochs: []),
                  train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]}
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
  """
  results = {'epochs': list(range(epochs)),
             'train_loss': [],
             'train_acc': [],
             'test_loss': [],
             'test_acc': []}

  for epoch in range(epochs):
    print(f"EPOCH [{epoch}]")
    train_loss, train_acc = train_step(model, train_data,
                                       loss, optimizer)
    
    test_loss, test_acc = test_step(model, test_data, loss)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% \n------------")
    results["train_loss"].append(train_loss.cpu())
    results["train_acc"].append(train_acc.cpu())
    results["test_loss"].append(test_loss.cpu())
    results["test_acc"].append(test_acc.cpu())

  return results