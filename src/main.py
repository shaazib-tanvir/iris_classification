from torch.utils.data.dataloader import DataLoader
from torch import nn
from iris import Iris
from neural_network import NeuralNetwork
import torch

def train(dataloader: DataLoader, model: NeuralNetwork, lossfn, optimizer):
    model.train()

    for batch, (x, y) in enumerate(dataloader):
        prediction = model(x)
        loss = lossfn(prediction, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), batch * batch_size + len(x)
        print(f"Loss: {loss}, Current: {current}")


def test(dataloader, model: NeuralNetwork, lossfn, device):
    model.eval()
    size = len(dataloader.dataset)
    correct = 0

    with torch.no_grad():
        for x, y in dataloader:
            prediction = model(x)
            correct += (prediction.argmax(1) == y.argmax(1)).type(torch.float32).sum().item()

    print(f"Accuracy: \033[32m{100 * correct / size}%\033[0m")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: \033[32m{device.upper()}\033[0m")

    train_dataset = Iris("data/iris.csv", train=True, device=device)
    test_dataset = Iris("data/iris.csv", train=False, device=device)
    batch_size = 64 

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = NeuralNetwork().to(device=device)
    lossfn = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 75

    for epoch in range(epochs):
        print()
        print(f"\033[34mEpoch\033[0m {epoch}")
        print("=" * 20)
        train(train_dataloader, model, lossfn, optimizer)
        test(test_dataloader, model, lossfn, device)
        print("=" * 20)


    print("\nExample Prediction: ")
    x, y = test_dataloader.dataset[0]
    x = x.reshape(1, len(x))
    prediction = model(x)
    predicted_species = train_dataset.get_species(prediction.argmax(1).item()) 
    actual_species = train_dataset.get_species(y.argmax().item())
    print(f"Predicted: {predicted_species}, Actual: {actual_species}")
