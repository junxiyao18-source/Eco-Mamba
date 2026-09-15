"""Animal Kingdom frame dataset used by Eco-Mamba."""

import csv
import os

import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .ecological_event_sampler import EcologicalEventSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VideoRecord:
    """Frame directory, frame count, and multi-label action targets for one video."""

    def __init__(self, path: str, num_frames: int, labels: list[int]):
        self.path = path
        self.num_frames = num_frames
        self.labels = labels


class AnimalKingdomDataset(Dataset):
    """Load Animal Kingdom clips with an ecological frame-selection policy."""

    def __init__(
        self,
        root: str,
        action_to_id: dict[str, int],
        num_frames: int = 12,
        sampling_strategy: str = "ees",
        transform=None,
        split: str = "train",
    ):
        self.root = root
        self.num_frames = num_frames
        self.transform = transform
        self.num_classes = len(action_to_id)
        self.annotation_path = os.path.join(root, "action_recognition", "annotation", f"{split}_light.csv")
        self.frame_sampler = EcologicalEventSampler(strategy=sampling_strategy)
        self.video_list, self.file_list = self._parse_annotations()

    def _parse_annotations(self):
        records, file_lists = [], []
        with open(self.annotation_path, newline="") as annotation_file:
            for row in csv.DictReader(annotation_file, delimiter=";"):
                video_id = row["video_id"]
                frame_directory = os.path.join(
                    self.root, "action_recognition", "dataset", "image", video_id
                )
                frame_names = sorted(os.listdir(frame_directory))
                labels = [int(label) for label in row["labels"].split(",") if label.strip()]
                records.append(VideoRecord(frame_directory, len(frame_names), labels))
                file_lists.append(frame_names)
        return records, file_lists

    @staticmethod
    def _load_image(directory: str, image_name: str) -> Image.Image:
        return Image.open(os.path.join(directory, image_name)).convert("RGB")

    def __getitem__(self, index: int):
        record = self.video_list[index]
        image_names = self.file_list[index]
        frame_paths = [os.path.join(record.path, image_name) for image_name in image_names]
        indices = self.frame_sampler.sample_indices(record.num_frames, self.num_frames, frame_paths)

        images, fallback_image = [], None
        for frame_index in indices:
            image_name = image_names[int(np.clip(frame_index, 0, len(image_names) - 1))]
            try:
                image = self._load_image(record.path, image_name)
                fallback_image = fallback_image or image
            except OSError:
                if fallback_image is None:
                    raise
                image = fallback_image.copy()
            images.append(image)

        clip = self.transform(images)
        clip = clip.view((self.num_frames, -1) + clip.size()[-2:])
        label = np.zeros(self.num_classes, dtype=np.float32)
        label[record.labels] = 1.0
        return clip, label

    def __len__(self) -> int:
        return len(self.video_list)
