'''
This script is for visualizing the images in the dataset.
'''
from gestalt_mnist import GestaltMNIST, GestaltFashionMNIST
import torch
from torchvision import transforms,utils
import matplotlib.pyplot as plt

test_loader = torch.utils.data.DataLoader(
        GestaltMNIST('./MNIST/proximity5/', transform=transforms.Compose([transforms.ToTensor(),])),batch_size=5, shuffle=False)
for image, label in test_loader:
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print(classes[label[0].int()])
    utils.save_image(image[0][0], f"proximity5.jpg")
    break