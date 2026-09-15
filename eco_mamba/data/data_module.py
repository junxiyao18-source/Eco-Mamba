"""Data loading and augmentation for the Animal Kingdom experiments."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms import Compose

from .animal_kingdom import AnimalKingdomDataset
from .video_transforms import (
    GroupCenterCrop, GroupGaussianBlur, GroupMultiScaleCrop, GroupNormalize,
    GroupRandomColorJitter, GroupRandomGrayscale, GroupRandomHorizontalFlip,
    GroupScale, GroupSolarization, Stack, ToTorchFormatTensor,
)


ANIMAL_KINGDOM_ACTIONS = {
    'Abseiling': 0, 'Attacking': 1, 'Attending': 2, 'Barking': 3, 'Being carried': 4, 'Being carried in mouth': 5, 'Being dragged': 6, 'Being eaten': 7, 'Biting': 8, 'Building nest': 9, 'Calling': 10, 'Camouflaging': 11, 'Carrying': 12, 'Carrying in mouth': 13, 'Chasing': 14, 'Chirping': 15, 'Climbing': 16, 'Coiling': 17, 'Competing for dominance': 18, 'Dancing': 19, 'Dancing on water': 20, 'Dead': 21, 'Defecating': 22, 'Defensive rearing': 23, 'Detaching as a parasite': 24, 'Digging': 25, 'Displaying defensive pose': 26, 'Disturbing another animal': 27, 'Diving': 28, 'Doing a back kick': 29, 'Doing a backward tilt': 30, 'Doing a chin dip': 31, 'Doing a face dip': 32, 'Doing a neck raise': 33, 'Doing a side tilt': 34, 'Doing push up': 35, 'Doing somersault': 36, 'Drifting': 37, 'Drinking': 38, 'Dying': 39, 'Eating': 40, 'Entering its nest': 41, 'Escaping': 42, 'Exiting cocoon': 43, 'Exiting nest': 44, 'Exploring': 45, 'Falling': 46, 'Fighting': 47, 'Flapping': 48, 'Flapping tail': 49, 'Flapping its ears': 50, 'Fleeing': 51, 'Flying': 52, 'Gasping for air': 53, 'Getting bullied': 54, 'Giving birth': 55, 'Giving off light': 56, 'Gliding': 57, 'Grooming': 58, 'Hanging': 59, 'Hatching': 60, 'Having a flehmen response': 61, 'Hissing': 62, 'Holding hands': 63, 'Hopping': 64, 'Hugging': 65, 'Immobilized': 66, 'Jumping': 67, 'Keeping still': 68, 'Landing': 69, 'Lying down': 70, 'Laying eggs': 71, 'Leaning': 72, 'Licking': 73, 'Lying on its side': 74, 'Lying on top': 75, 'Manipulating object': 76, 'Molting': 77, 'Moving': 78, 'Panting': 79, 'Pecking': 80, 'Performing sexual display': 81, 'Performing allo-grooming': 82, 'Performing allo-preening': 83, 'Performing copulatory mounting': 84, 'Performing sexual exploration': 85, 'Performing sexual pursuit': 86, 'Playing': 87, 'Playing dead': 88, 'Pounding': 89, 'Preening': 90, 'Preying': 91, 'Puffing its throat': 92, 'Pulling': 93, 'Rattling': 94, 'Resting': 95, 'Retaliating': 96, 'Retreating': 97, 'Rolling': 98, 'Rubbing its head': 99, 'Running': 100, 'Running on water': 101, 'Sensing': 102, 'Shaking': 103, 'Shaking head': 104, 'Sharing food': 105, 'Showing affection': 106, 'Sinking': 107, 'Sitting': 108, 'Sleeping': 109, 'Sleeping in its nest': 110, 'Spitting': 111, 'Spitting venom': 112, 'Spreading': 113, 'Spreading wings': 114, 'Squatting': 115, 'Standing': 116, 'Standing in alert': 117, 'Startled': 118, 'Stinging': 119, 'Struggling': 120, 'Surfacing': 121, 'Swaying': 122, 'Swimming': 123, 'Swimming in circles': 124, 'Swinging': 125, 'Tail swishing': 126, 'Trapped': 127, 'Turning around': 128, 'Undergoing chrysalis': 129, 'Unmounting': 130, 'Unrolling': 131, 'Urinating': 132, 'Walking': 133, 'Walking on water': 134, 'Washing': 135, 'Waving': 136, 'Wrapping itself around prey': 137, 'Wrapping prey': 138, 'Yawning': 139,
}


class AnimalKingdomDataModule:
    """Construct reproducible Eco-Mamba data loaders for Animal Kingdom."""

    def __init__(self, args, dataset_root: str):
        if args.dataset != "animalkingdom":
            raise ValueError("This compact repository supports only --dataset animalkingdom.")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.dataset_root = dataset_root
        self.num_frames = args.num_frames
        self.sampling_strategy = args.sampling_strategy
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    @staticmethod
    def action_to_id() -> dict[str, int]:
        return ANIMAL_KINGDOM_ACTIONS

    def train_transform(self):
        mean, std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
        return Compose([GroupMultiScaleCrop(224, [1, .875, .75, .66]), GroupRandomHorizontalFlip(True), GroupRandomColorJitter(p=.8, brightness=.4, contrast=.4, saturation=.2, hue=.1), GroupRandomGrayscale(p=.2), GroupGaussianBlur(p=0.), GroupSolarization(p=0.), Stack(roll=False), ToTorchFormatTensor(div=True), GroupNormalize(mean, std)])

    @staticmethod
    def evaluation_transform():
        mean, std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
        return Compose([GroupScale(256), GroupCenterCrop(224), Stack(roll=False), ToTorchFormatTensor(div=True), GroupNormalize(mean, std)])

    def _dataset(self, split: str, transform):
        return AnimalKingdomDataset(self.dataset_root, self.action_to_id(), self.num_frames, self.sampling_strategy, transform=transform, split=split)

    def train_loader(self):
        dataset = self._dataset("train", self.train_transform())
        return DataLoader(dataset, batch_size=self.batch_size, sampler=RandomSampler(dataset, num_samples=2500), num_workers=self.num_workers, pin_memory=True)

    def validation_loader(self):
        dataset = self._dataset("val", self.evaluation_transform())
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
