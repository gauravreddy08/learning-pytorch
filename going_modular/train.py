print(f"[INFO] Installing Dependencies...")
import argparse
import pip
parser = argparse.ArgumentParser()

parser.add_argument("--train", type=str)
parser.add_argument("--test", type=str)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--hidden", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.1)

args = parser.parse_args()

import os
import torch 
from torch import nn
from torchvision import transforms

from datetime import datetime

try: 
  import torchinfo
except:
  pip.main(['install', 'torchinfo'])       
  import torchinfo

import data_setup, engine, models
from torchinfo import summary

EPOCHS = args.epochs
BATCH_SIZE = args.batch
HIDDEN_UNITS = args.hidden
LEARNING_RATE = args.lr
LINE_BR = "-"*90

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('\n')
print(LINE_BR)
print(f"[Hyperparameters]: ")
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Hidden Units: {HIDDEN_UNITS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"CUDA Available = {torch.cuda.is_available()} | Device: {DEVICE}")
print(LINE_BR)

train_dir = args.train
test_dir = args.test

print(f"[INFO] Training Directory: {train_dir}")
print(f"[INFO] Testing Directory: {test_dir}")
print(LINE_BR)

data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataset(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

print(f"[INFO] Model Initialised...")

model = models.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(DEVICE)

summary(model, input_size=[1, 3, 64, 64])

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)
print(LINE_BR)
print(f"[INFO] Training...")
# Start training with help from engine.py
engine.train(model=model,
             train_data=train_dataloader,
             test_data=test_dataloader,
             loss=loss_fn,
             optimizer=optimizer,
             epochs=EPOCHS,
             device=DEVICE)

name = "model_" + datetime.now().strftime("%d%M%y_%H%M%S")+ '.pth'
torch.save(obj=model.state_dict(), f=name)
print(f"[INFO] Saved model as {name}")
print(LINE_BR)
