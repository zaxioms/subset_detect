import numpy as np
import plotly.express as px
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

from models.bbox import bbox
from utils.dataset import BallDataset

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("cuda")
else:
    device = torch.device("cpu")
    print("cpu")

EPOCHS = 50
BATCH_SIZE = 32
loss_arr = []
bd = BallDataset(
    csv_file="data/true_annotations/resized_annotations.csv",
    root_dir="data/resized_images",
    transform=transforms.Compose([transforms.ToTensor()]),
)
train, test = torch.utils.data.random_split(bd, [3500, len(bd) - 3500])

trainset = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
testset = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

net = bbox().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
for epoch in tqdm(range(EPOCHS)):
    for data in tqdm(trainset):
        X, y = data
        X, y = X.to(device), y.to(device)
        net.zero_grad()
        output = net(X)
        loss = F.l1_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)
    loss_arr.append(loss.item())

x = np.linspace(1, EPOCHS, EPOCHS)
y = loss_arr

fig = px.line(x=x, y=y)
fig.show()
