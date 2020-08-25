import torch
from matplotlib import pyplot as plt
torch.set_printoptions(profile="full")

dataset = torch.load('./MNIST/half/test.pt')
print(dataset[0][0])