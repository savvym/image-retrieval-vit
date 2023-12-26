from abc import abstractmethod
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageOps


class BaseDataset(Dataset):

    def __init__(self, data_dir, input_size=224, split="train"):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.input_size = input_size
        is_training = split == "train"
        self.transform = build_transform(input_size, is_training)
        self.set_paths_and_labels()

    @abstractmethod
    def set_paths_and_labels(self):
        pass

    def __len__(self,):
        return len(self.paths)

    def __getitem__(self, idx):
        pth = self.paths[idx]
        img = Image.open(pth).convert('RGB')
        if img.size[0] < self.input_size or img.size[1] < self.input_size:
        # 计算需要添加的填充量
            left_padding = max(0, self.input_size - img.size[0])
            top_padding = max(0, self.input_size - img.size[1])
            # 随机选择填充颜色
            fill_color = (0, 0, 0)
            # 添加填充
            img = ImageOps.expand(img, border=(left_padding, top_padding), fill=fill_color)
        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

    def __repr__(self,):
        return f"{self.__class__.__name__}(split={self.split}, len={len(self)})"


def build_transform(input_size=224, is_training=False):
    t = []
    t.append(transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC))
    if is_training:
        t.append(transforms.RandomCrop(input_size))
        t.append(transforms.RandomHorizontalFlip(0.5))
    else:
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))),
    return transforms.Compose(t)


