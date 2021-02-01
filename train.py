import numpy as np
import plotly.express as px
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from tqdm import tqdm
from models.bbox import bbox
from utils.dataset import BallDataset
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Device:", torch.cuda.get_device_name(0))
    print("Cuda Version:", torch.version.cuda)
else:
    device = torch.device("cpu")
    print("cpu")


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    loader = transforms.Compose([transforms.Scale(256), transforms.ToTensor()])
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    print("Image tensor shape: ", image.shape)
    return image.cuda()  # assumes that you're using GPU


EPOCHS = 5
BATCH_SIZE = 32
loss_arr = []
bd = BallDataset(
    csv_file="data/combined_data/combined.csv",
    root_dir="data/combined_data/combined_images",
    transform=transforms.Compose([transforms.ToTensor()]),
)

train, test = torch.utils.data.random_split(bd, [10000, len(bd) - 10000])

trainset = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

net = bbox().to(device)
torch.cuda.empty_cache()
optimizer = optim.Adam(net.parameters(), lr=0.001)
for epoch in tqdm(range(EPOCHS)):
    for data in tqdm(trainset):
        X, y = data
        X, y = X.to(device), y.to(device)
        net.zero_grad()
        output = net(X)
        # print(output)
        loss = F.l1_loss(output, y)
        loss.backward()
        optimizer.step()
    print("Loss: ", loss.item())
    loss_arr.append(loss.item())

x_axis = np.linspace(1, EPOCHS, EPOCHS)
y_axis = loss_arr

fig = px.line(x=x_axis, y=y_axis)
fig.show()

net.eval()
with torch.no_grad():
    out = net(image_loader("data/resized_images/3375.jpg"))
print("Test BBox: ", out)
fig, ax = plt.subplots(1)
plt.imshow(test)
rect = patches.Rectangle(
    (out[0][0], out[0][1]),
    out[0][2],
    out[0][3],
    linewidth=1,
    edgecolor="r",
    facecolor="none",
)
ax.add_patch(rect)
plt.show()
