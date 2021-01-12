import pdb
from typing import Union

import torch
from torchvision.utils import save_image

from lisa import LISA
from tqdm import tqdm


class Translator:
    def __init__(self):
        self.signs = ['stop', 'speedLimitUrdbl', 'speedLimit25', 'pedestrianCrossing', 'speedLimit35', 'turnLeft',
                      'slow', 'speedLimit15', 'speedLimit45', 'rightLaneMustTurn', 'signalAhead', 'keepRight',
                      'laneEnds', 'school', 'merge', 'addedLane', 'rampSpeedAdvisory40', 'rampSpeedAdvisory45',
                      'curveRight', 'speedLimit65', 'truckSpeedLimit55', 'thruMergeLeft', 'speedLimit30', 'stopAhead',
                      'yield', 'thruMergeRight', 'dip', 'schoolSpeedLimit25', 'thruTrafficMergeLeft', 'noRightTurn',
                      'rampSpeedAdvisory35', 'curveLeft', 'rampSpeedAdvisory20', 'noLeftTurn', 'zoneAhead25',
                      'zoneAhead45', 'doNotEnter', 'yieldAhead', 'roundabout', 'turnRight', 'speedLimit50',
                      'rampSpeedAdvisoryUrdbl', 'rampSpeedAdvisory50', 'speedLimit40', 'speedLimit55', 'doNotPass',
                      'intersection']
        warnings = ['pedestrianCrossing', 'slow', 'signalAhead', 'laneEnds', 'school', 'merge', 'addedLane',
                    'turnLeft', 'thruMergeRight', 'thruMergeLeft', 'dip', 'thruTrafficMergeLeft', 'roundabout',
                    'intersection', 'curveRight', 'curveLeft', 'turnRight', ]
        self.warnings = [w.lower() for w in warnings]
        self.easy_map = {
            'ahead': 'warning',
            'stop': 'stop',
            'yield': 'yield',
            'speedLimit': 'speed',
            'zoneAhead': 'speed',
        }
        self.reverse_map = ['stop', 'yield', 'speed', 'warning', 'regulatory']
        self.numerical_map = {key: i for i, key in enumerate(self.reverse_map)}

    def translate(self, name: str) -> str:
        for n, o in self.easy_map.items():
            if n.lower() in name.lower():
                return o
        if 'speed' in name.lower():
            return 'warning'
        if name.lower() in self.warnings:
            return 'warning'
        return 'regulatory'

    def __call__(self, y: Union[int, torch.tensor]) -> int:
        return self.numerical_map[self.translate(self.signs[y])]


def main():
    translator = Translator()
    dataset = LISA(download=True, root='./data', train=True, target_transform=translator)
    for name, i in translator.numerical_map.items():
        images = [img for img, y in dataset if y == i]
        images = torch.stack(images)
        indices = torch.randperm(len(images))
        images = images[indices[:100]]
        save_image(images, f'examples/png/{name}.png', nrow=10)


if __name__ == '__main__':
    main()
