import os.path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from PIL import Image

from .utils import download_and_extract_archive
from .vision import VisionDataset


_TARBALL_URL = (
    "https://www.robots.ox.ac.uk/~vgg/data/pose_evaluation/PoseEvaluatorDataset_files/poseevaluator_dataset.tgz"
)


_TARBALL_MD5SUM = "f94c2a389f8df2f7f1b3e8946322f401"


IMAGE_SETS = [
    "4weds",
    "apollo13",
    "boy",
    "buffy_s5e2",
    "buffy_s5e3",
    "forrest",
    "gandhi",
    "Graduate",
    "GroundhogDay",
    "love",
    "notting",
    "Witness",
]


class _StickCoordinates(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float


class _StickmenAnnotation(NamedTuple):
    torso: _StickCoordinates
    left_upper_arm: _StickCoordinates
    right_upper_arm: _StickCoordinates
    left_lower_arm: _StickCoordinates
    right_lower_arm: _StickCoordinates
    head: _StickCoordinates


def _parse_sticks_standard(fpath: str) -> Dict[str, List[_StickmenAnnotation]]:
    """Read annotations; returns map of filename to stickmen annotations."""
    all_annotations: Dict[str, List[_StickmenAnnotation]] = {}
    with open(fpath, "r") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            fname, num_ann_str = line.split(sep=" ")
            assert fname not in all_annotations

            frame_annotations: List[_StickmenAnnotation] = []
            for i in range(int(num_ann_str)):
                annotation = []
                for j in range(6):
                    annotation.append(_StickCoordinates(*[float(x) for x in fh.readline().split(sep=" ", maxsplit=3)]))
                frame_annotations.append(_StickmenAnnotation(*annotation))
            all_annotations[fname] = frame_annotations
    return all_annotations


class HumanPoseEvaluator(VisionDataset):
    """`Human Pose Evaluator <https://www.robots.ox.ac.uk/~vgg/data/pose_evaluation/>`_.

    This dataset is derived from Hollywood movies and "Buffy, the
    vampire slayer" series.  The images are randomly sampled from the
    aforementioned videos.  All humans with more than three parts
    visible are annotated with upper body stickmen (6 parts: head,
    torso, upper and lower arms).

    Args:
        root (string): Root directory of dataset.
        image_sets (list of strings, optional): The list of image sets
            to load.  The image sets are the different videos that
            make up the dataset.  The full list is in
            `torchvision.datasets.human_pose_evaluator.IMAGE_SETS`.
            Defaults to `None` which loads all.
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """

    def __init__(
        self,
        root: str,
        image_sets: Optional[List[str]] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(
            root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        base_folder = "poseevaluator_dataset"
        self._images_folder = os.path.join(base_folder, "images")
        self._anns_folder = os.path.join(base_folder, "data")
        if image_sets:
            for imgset in image_sets:
                if imgset not in IMAGE_SETS:
                    raise ValueError("invalid image set '%s' (needs to be one of %s)" % (imgset, IMAGE_SETS))
            self._image_sets = image_sets
        else:
            self._image_sets = IMAGE_SETS

        if download:
            self._download()
        elif not self._data_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._idx_to_data: List[Tuple[str, List[_StickmenAnnotation]]] = []
        for imgset in self._image_sets:
            sticks_standard_fpath = os.path.join(self.root, self._anns_folder, imgset, imgset + "_sticks_standard.txt")
            sticks = _parse_sticks_standard(sticks_standard_fpath)

            image_list_fpath = os.path.join(self.root, self._images_folder, imgset, "images.list")
            with open(image_list_fpath, "r") as fh:
                for line in fh:
                    assert line.endswith(".jpg\n")
                    basename = line[:-1]
                    # Not all images have annotated stickmen and those
                    # that don't do not appear in the sticks_standard
                    # file, hence the `dict.get(key, default=[])`
                    self._idx_to_data.append((
                        os.path.join(imgset, basename),
                        sticks.get(basename, []),
                    ))

    def __len__(self) -> int:
        return len(self._idx_to_data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        relpath, target = self._idx_to_data[idx]
        image = Image.open(os.path.join(self.root, self._images_folder, relpath))
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target

    def extra_repr(self) -> str:
        return "Image Sets: %s" % self._image_sets

    def _data_exists(self) -> bool:
        if not os.path.isdir(os.path.join(self.root, self._images_folder)) or not os.path.isdir(
            os.path.join(self.root, self._anns_folder)
        ):
            return False
        else:
            return True

    def _download(self) -> None:
        if self._data_exists():
            return

        download_and_extract_archive(
            _TARBALL_URL,
            download_root=str(self.root),
            md5=_TARBALL_MD5SUM,
        )
