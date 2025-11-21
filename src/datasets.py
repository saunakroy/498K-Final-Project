import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

'''
Loads NERF synthetic dataset from given path.
- Images from train directory in PNG format
- camera poses and intrinsics from transforms_train.json
- returns (image, caption, pose)
- Necessary for training text-conditioned Gaussian Splatting model and fusion cross-attention module
'''
class ImageDataset(Dataset):

    def __init__(self, root_dir, split="train", caption=None, image_size=256):
        """
        root_dir: path to dataset (e.g., /content/.../data/hotdog)
        split: "train" or "test"
        caption: text caption used for ALL images (can be replaced later)
        image_size: resize images to this size
        """
        self.root_dir = root_dir
        self.split = split

        # Default caption if none is given
        self.caption = caption or "a synthetic 3D scene"

        # Load transforms JSON
        transform_json = os.path.join(root_dir, f"transforms_{split}.json")
        with open(transform_json, "r") as f:
            meta = json.load(f)

        self.frames = meta["frames"]
        self.camera_angle_x = meta.get("camera_angle_x", None)

        # Preprocess transform for images
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        image_path = os.path.join(self.root_dir, frame["file_path"] + ".png")

        # Load image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Pose matrix 4x4
        pose = torch.tensor(frame["transform_matrix"], dtype=torch.float32)

        return {
            "image": image,              # [3, H, W]
            "caption": self.caption,     # string
            "pose": pose                 # [4, 4]
        }