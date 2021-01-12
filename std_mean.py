from itertools import chain
from lisa import LISA
import torch


def main():
    train = LISA(root='./data', download=True, train=True)
    test = LISA(root='./data', download=True, train=False)
    data = [img for img, y in chain(train, test)]
    both = torch.stack(data)
    print(both.mean(dim=(0, 2, 3)))
    print(both.std(dim=(0, 2, 3)))


"""
mean = tensor([0.4563, 0.4076, 0.3895])
std = tensor([0.2298, 0.2144, 0.2259])
"""

if __name__ == '__main__':
    main()
