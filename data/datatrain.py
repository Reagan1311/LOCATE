import os
import torch
import random
from PIL import Image
from torch.utils import data
from torchvision import transforms


class TrainData(data.Dataset):
    def __init__(self, exocentric_root, egocentric_root, resize_size=256, crop_size=224, divide="Seen"):

        self.exocentric_root = exocentric_root
        self.egocentric_root = egocentric_root

        self.image_list = []
        self.exo_image_list = []
        self.resize_size = resize_size
        self.crop_size = crop_size
        if divide == "Seen":
            self.aff_list = ['beat', "boxing", "brush_with", "carry", "catch", "cut", "cut_with", "drag", 'drink_with',
                             "eat", "hit", "hold", "jump", "kick", "lie_on", "lift", "look_out", "open", "pack", "peel",
                             "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick", "stir", "swing", "take_photo",
                             "talk_on", "text_on", "throw", "type_on", "wash", "write"]
            self.obj_list = ['apple', 'axe', 'badminton_racket', 'banana', 'baseball', 'baseball_bat',
                             'basketball', 'bed', 'bench', 'bicycle', 'binoculars', 'book', 'bottle',
                             'bowl', 'broccoli', 'camera', 'carrot', 'cell_phone', 'chair', 'couch',
                             'cup', 'discus', 'drum', 'fork', 'frisbee', 'golf_clubs', 'hammer', 'hot_dog',
                             'javelin', 'keyboard', 'knife', 'laptop', 'microwave', 'motorcycle', 'orange',
                             'oven', 'pen', 'punching_bag', 'refrigerator', 'rugby_ball', 'scissors',
                             'skateboard', 'skis', 'snowboard', 'soccer_ball', 'suitcase', 'surfboard',
                             'tennis_racket', 'toothbrush', 'wine_glass']
        else:
            self.aff_list = ["carry", "catch", "cut", "cut_with", 'drink_with',
                             "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
                             "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
                             "swing", "take_photo", "throw", "type_on", "wash"]
            self.obj_list = ['apple', 'axe', 'badminton_racket', 'banana', 'baseball', 'baseball_bat',
                             'basketball', 'bed', 'bench', 'bicycle', 'binoculars', 'book', 'bottle',
                             'bowl', 'broccoli', 'camera', 'carrot', 'cell_phone', 'chair', 'couch',
                             'cup', 'discus', 'drum', 'fork', 'frisbee', 'golf_clubs', 'hammer', 'hot_dog',
                             'javelin', 'keyboard', 'knife', 'laptop', 'microwave', 'motorcycle', 'orange',
                             'oven', 'pen', 'punching_bag', 'refrigerator', 'rugby_ball', 'scissors',
                             'skateboard', 'skis', 'snowboard', 'soccer_ball', 'suitcase', 'surfboard',
                             'tennis_racket', 'toothbrush', 'wine_glass']

        self.transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))])

        # image list for egocentric images
        files = os.listdir(self.exocentric_root)
        for file in files:
            file_path = os.path.join(self.exocentric_root, file)
            obj_files = os.listdir(file_path)
            for obj_file in obj_files:
                obj_file_path = os.path.join(file_path, obj_file)
                images = os.listdir(obj_file_path)
                for img in images:
                    img_path = os.path.join(obj_file_path, img)
                    self.image_list.append(img_path)

        # multiple affordance labels for exo-centric samples

    def __getitem__(self, item):

        # load egocentric image
        exocentric_image_path = self.image_list[item]
        names = exocentric_image_path.split("/")
        aff_name, object = names[-3], names[-2]
        exocentric_image = self.load_img(exocentric_image_path)
        aff_label = self.aff_list.index(aff_name)

        ego_path = os.path.join(self.egocentric_root, aff_name, object)
        obj_images = os.listdir(ego_path)
        idx = random.randint(0, len(obj_images) - 1)
        egocentric_image_path = os.path.join(ego_path, obj_images[idx])
        egocentric_image = self.load_img(egocentric_image_path)

        # pick one available affordance, and then choose & load exo-centric images
        num_exo = 3
        exo_dir = os.path.dirname(exocentric_image_path)
        exocentrics = os.listdir(exo_dir)
        exo_img_name = [os.path.basename(exocentric_image_path)]
        exocentric_images = [exocentric_image]
        # exocentric_labels = []

        if len(exocentrics) > num_exo:
            for i in range(num_exo - 1):
                exo_img_ = random.choice(exocentrics)
                while exo_img_ in exo_img_name:
                    exo_img_ = random.choice(exocentrics)
                exo_img_name.append(exo_img_)
                tmp_exo = self.load_img(os.path.join(exo_dir, exo_img_))
                exocentric_images.append(tmp_exo)
        else:
            for i in range(num_exo - 1):
                exo_img_ = random.choice(exocentrics)
                # while exo_img_ in exo_img_name:
                #     exo_img_ = random.choice(exocentrics)
                exo_img_name.append(exo_img_)
                tmp_exo = self.load_img(os.path.join(exo_dir, exo_img_))
                exocentric_images.append(tmp_exo)

        exocentric_images = torch.stack(exocentric_images, dim=0)  # n x 3 x 224 x 224

        return exocentric_images, egocentric_image, aff_label

    def load_img(self, path):
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):

        return len(self.image_list)
