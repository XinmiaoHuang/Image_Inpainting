import argparse


class BaseParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument("--checkpoint", type=str, default='./model/_gnet.pth')
        self.parser.add_argument("--batch_size", type=int, default=6)
        self.parser.add_argument("--test_bsize", type=int, default=6)
        self.parser.add_argument("--img_size", type=tuple, default=(256, 256))
        self.parser.add_argument("--input_nc", type=int, default=6)
        self.parser.add_argument("--epochs", type=int, default=150)
        self.parser.add_argument("--steps", type=int, default=10)
        self.parser.add_argument("--save_per_iter", type=int, default=600)
        self.parser.add_argument("--lr", type=float, default=0.0002)
        self.parser.add_argument("--data_dir", type=str, default='/media/homee/Data/Dataset/coco/train2014')
        self.parser.add_argument("--mask_dir", type=str, default='./random_mask/mask_slim/')

    def parse(self):
        return self.parser.parse_args()
