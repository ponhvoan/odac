import os
from typing import Optional
from dalib.vision.datasets import ImageList
from dalib.vision.datasets._util import download as download_data, check_exits

class VisDAOW(ImageList):
    """`VisDA-2017 <http://ai.bu.edu/visda-2017/assets/attachments/VisDA_2017.pdf>`_ Dataset

    Parameters:
        - **root** (str): Root directory of dataset
        - **task** (str): The task (domain) to create dataset. Choices include ``'T'``: training and ``'V'``: validation.
        - **download** (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        - **transform** (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, ``transforms.RandomCrop``.
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            train/
                aeroplance/
                    *.png
                    ...
            validation/
            image_list/
                train.txt
                validation.txt
    """
    download_list = [
        ("image_list", "image_list.txt", "https://drive.google.com/uc?id=1t3c3QNSF5t9clCwsCzK5uc1td64HgA6l&export=download"),
        ("train", "train.tar", "http://csr.bu.edu/ftp/visda17/clf/train.tar"),
        ("validation", "validation.tar", "http://csr.bu.edu/ftp/visda17/clf/validation.tar")
    ]
    image_list = {
        "T": "image_list/train.txt",
        "V": "image_list/validation.txt"
    }
    CLASSES = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
               'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']

    def __init__(self, root: str, task: str, labeled_num: int, download: Optional[bool] = False,**kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(VisDAOW, self).__init__(root, VisDAOW.CLASSES[:labeled_num], data_list_file=data_list_file, **kwargs)

    
