import os

import cv2
import torch.utils.data as data


class CrackDataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None):
        assert os.path.exists(root), f"path {root} does not exist."
        if train:
            self.image_root = os.path.join(root, "dataset", "CRACK-TR", "images")
            self.mask_root = os.path.join(root, "dataset", "CRACK-TR", "masks")

        else:
            self.image_root = os.path.join(root, "dataset", "CRACK-TE", "images")
            self.mask_root = os.path.join(root, "dataset", "CRACK-TE", "masks")
            assert os.path.exists(self.image_root), f"path '{self.image_root}' does not exist."
            assert os.path.exists(self.mask_root), f"path '{self.mask_root}' does not exist."

        image_names = [p for p in os.listdir(self.image_root) if p.endswith(".jpg")]
        mask_names = [p for p in os.listdir(self.mask_root) if p.endswith(".jpg")]
        assert len(image_names) > 0, f"not find any images in {self.image_root}."

        # check image and mask
        re_mask_names = []
        for p in image_names:
            mask_name = p
            assert mask_name in mask_names, f"{p} has no corresponding mask."
            re_mask_names.append(mask_name)
        mask_names = re_mask_names

        self.images_path = [os.path.join(self.image_root, n) for n in image_names]
        self.masks_path = [os.path.join(self.mask_root, n) for n in mask_names]

        self.transforms = transforms

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]
        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        assert image is not None, f"failed to read image: {image_path}."
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        h, w, _ = image.shape

        target = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        assert target is not None, f"failed to read mask: {mask_path}."

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_images = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)

        return batched_images, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_images = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_images):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_images


if __name__ == '__main__':
    train_dataset = CrackDataset("../", train=True)
    print(len(train_dataset))

    val_dataset = CrackDataset("../", train=False)
    print(len(val_dataset))

    i, t = train_dataset[0]
    print(i.shape)
    print(t.shape)
