'''
This script is for visualizing the images in the dataset.
'''
from gestalt_mnist import GestaltMNIST, GestaltFashionMNIST
import torch
from torchvision import transforms,utils
import matplotlib.pyplot as plt


trans = ['original', 'reverse', 'half', 'quarter',
         'closure2', 'continuity2', 'illusory2', 'illusory_complex_2_2']

for tran in trans:
        test_loader = torch.utils.data.DataLoader(GestaltMNIST(f'./FashionMNIST/{tran}/', transform=transforms.Compose([transforms.ToTensor(),])),batch_size=100, shuffle=False)
        n = 0
        for image, label in test_loader:
                classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                        'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                print(classes[label[0].int()])
                l = []
                for j in range(50):
                        for i in range(len(image)):
                                if label[i] == len(l):
                                        print(label[i])
                                        l.append(image[i][0])
                                        break
                images1 = torch.cat(l[0:5], 1)
                images2 = torch.cat(l[5:], 1)
                images = torch.cat((images1, images2), 0)
                utils.save_image(images, f"Fashion{tran}_{n}.png")
                n+=1
                if n == 10:
                        break
