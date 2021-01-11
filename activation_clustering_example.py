from torch.utils.data.dataset import Dataset

from lisa import LISA


def main():
    dataset = LISA(download=True, root='./data')


if __name__ == '__main__':
    main()
