import argparse

from src.Setup import Setup
from src.Trainer import Trainer


class Demo:
    def __init__(self):
        self.setup = Setup

    def run(self, creat_dataset, dataset_dir):
        self.setup.run(creat_dataset, dataset_dir)


classes = ['Pain', 'Neutral']


def main():
    # Flags:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, type=bool, help='train pain mode?')
    parser.add_argument('--camera', default=False, type=bool, help='is camera mode?')
    parser.add_argument('--create_dataset', default=False, type=bool, help='create your own dataset?')
    parser.add_argument('--dataset_dir', default="./dataset/", type=str, help='path to the dataset')

    args = parser.parse_args()

    if args.train:
        train_path = [args.dataset_dir + c + '/train/*.png' for c in classes]
        test_path = [args.dataset_dir + c + '/test/*.png' for c in classes]
        model = Trainer(classes, train_path)
        model.train("svm")  # knn or svm
        model.test(test_path)

    if args.camera:
        Demo().run(False, None)

    if args.create_dataset:
        Demo().run(True, args.dataset_dir)


if __name__ == '__main__':
    main()
