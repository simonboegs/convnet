import torch
import torch.nn as nn
import numpy as np
from models import Model1
from utils import create_dataloaders

LR = .003
PRINT_INTERVAL = 100
WEIGHTS_SAVE = "saves/model.pth"

def train(model, train_dl, test_dl, epochs, tb_writer=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):    
        print(f"epoch {epoch+1}-----")
        running_loss = 0.0
        last_loss = 0.0
        for batch_idx, data in enumerate(train_dl):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % PRINT_INTERVAL == PRINT_INTERVAL - 1: 
                last_loss = running_loss / PRINT_INTERVAL
                print(f" batch {i+1}/{len(train_dl)} loss: {last_loss}")
                if tb_writer is not None:
                    writer_idx = epoch_idx * len(train_dl) + batch_idx + 1
                    tb_writer.add_scalar("Loss/train", last_loss, writer_idx)
                running_loss = 0
    print("training finished")
    torch.save(model.state_dict(), WEIGHTS_SAVE) 
    print(f"model saved -> {WEIGHTS_SAVE}")


if __name__ == "__main__":
    model = Model1()
