import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST, FashionMNIST
import os

class GestaltMNIST(MNIST):
    def __init__(self, root, transform=None, target_transform=None):
        self._processed_folder = root
        MNIST.__init__(self, root, train=False, transform=transform, target_transform=None,
                 download=False)


    @property
    def processed_folder(self):
        return self._processed_folder

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

class GestaltFashionMNIST(FashionMNIST):
    def __init__(self, root, transform=None, target_transform=None):
        self._processed_folder = root
        FashionMNIST.__init__(self, root, train=False, transform=transform, target_transform=None,
                 download=False)


    @property
    def processed_folder(self):
        return self._processed_folder

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))


if __name__ == '__main__':
    test_loader = torch.utils.data.DataLoader(
        GestaltFashionMNIST('./FashionMNIST/original/', transform=transforms.Compose([transforms.ToTensor(),])),batch_size=5, shuffle=True)
    for image, label in test_loader:
        print(label[0])
        plt.imshow(image[0][0])
        plt.show()