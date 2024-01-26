# adapted from:
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py


import os
import collections
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision.datasets.vision import VisionDataset
import xml.etree.ElementTree as ET

from references.detection.transforms import Compose, RandomHorizontalFlip, ToTensor
from references.detection.utils import collate_fn


CLASSES = ['__background__', 'dog']


class FaceDetection(VisionDataset):
    """
    Dataset links:
        PASCAL VOC 2010: http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
        PASCAL-Part annotations: http://roozbehm.info/pascal-parts/pascal-parts.html
    Args:
        root (string): Root directory of the Pascal Part Dataset. Must contain the fololowing dir structure:
            Images: `root`/JPEGImages/*.jpg
            Object and Part annotations: `root`/Annotations_Part_json/*.json [see `parse_Pascal_VOC_Part_Anno.py`]
            train/val splits: `root`/ImageSets/Main/`image_set`.txt
            class2ind_file: `root`/Classes/`class2ind_file`.txt
        image_set (string, optional): Select the image_set to use, e.g. train (default), trainval, val
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, required): A function/transform that takes in the target and transforms it.
        class2ind_file: file containing list of class names and class index that are to be considered from all annotations.
            Other object/part classes will be ignored.
            Default: `face_semantic_class2ind`.
            Note: `__background__` class should also be present.
        use_objects: if True (default: True), use object annotations.
        use_parts: if True (default: False), use part annotations that are present inside an object.
        return_separate_targets: if True, return img, obj_target, part_target instead of img, target (default: False)
            should be set True only for training JointDetector.
        part_class2ind_file: similar to `class2ind_file` but will have part classes (default: None).
            should be provided only if return_separate_targets=True otherwise should be provided as `class2ind_file`.
    """
    def __init__(self, root, image_set='train', transforms=None, transform=None, target_transform=None, class2ind_file='face_semantic_class2ind',
                 use_objects=True, use_parts=False, return_separate_targets=False, part_class2ind_file=None):
        super(FaceDetection, self).__init__(root, transforms, transform, target_transform)
        
        '''
        % root = '/home/nano01/a/chowdh23/Part_Detector/animal_part_det'
        '''
        
        image_dir = '%s/JPEG_images/' % root
        annotation_dir = '%s/Annotation/' % root
        splits_file = '%s/ImageSets/%s.txt' % (root, image_set)
        class2ind_file = '%s/classes/%s.txt' % (root, class2ind_file)

        if not os.path.isdir(image_dir) or not os.path.isdir(annotation_dir) or not os.path.exists(splits_file) or not os.path.exists(class2ind_file):
            raise RuntimeError('Dataset not found or corrupted.')
        if not use_objects and not use_parts:
            raise RuntimeError('Atleast 1 of objects and parts have to be used')
        self.use_objects = use_objects
        self.use_parts = use_parts
        self.return_separate_targets = return_separate_targets
        
        class2ind_list = np.loadtxt(class2ind_file, dtype=str) # shape [n_classes, 2]
        self.class2ind = {k: int(v) for k, v in class2ind_list}
        self.classes = set(self.class2ind.keys())
        self.n_classes = len(np.unique(list(self.class2ind.values())))

        # need to understand this 
        if self.return_separate_targets:
            part_class2ind_file = '%s/Classes/%s.txt' % (root, part_class2ind_file)
            if not os.path.exists(part_class2ind_file):
                raise RuntimeError('For separate targets, class2ind_file is for objects and part_class2ind_file is for parts')
            class2ind_part_list = np.loadtxt(part_class2ind_file, dtype=str) # shape [n_classes, 2]
            self.part_class2ind = {k: int(v) for k, v in class2ind_part_list}
            self.part_classes = set(self.part_class2ind.keys())
            self.part_n_classes = len(np.unique(list(self.part_class2ind.values())))
        else:
            self.part_class2ind = self.class2ind
            self.part_classes = self.classes
        
        file_names = np.loadtxt(splits_file, dtype=str)
        self.images = ['%s/%s.jpg' % (image_dir, x) for x in file_names]
        self.annotations = ['%s/%s.xml' % (annotation_dir, x) for x in file_names]
        
        print('Image Set: %s,  Samples: %d, Objects: %s, Parts: %s, separate_targets: %s' % (image_set, len(self.images), use_objects, use_parts,
                                                                                             self.return_separate_targets))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
#        print(self.images[index])
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())
        boxes, labels, iscrowd = [], [], []
        
        if not isinstance(target['annotation']['object'], list):
            target['annotation']['object'] = [target['annotation']['object']]
        for obj in target['annotation']['object']:
            xmin = int(obj['bndbox']['xmin']) - 1
            ymin = int(obj['bndbox']['ymin']) - 1
            xmax = int(obj['bndbox']['xmax']) - 1
            ymax = int(obj['bndbox']['ymax']) - 1
            boxes.append([xmin, ymin, xmax, ymax])    
            labels.append(self.class2ind[obj['name']])
            iscrowd.append(bool(int(obj['difficult'])))
        
        boxes = torch.Tensor(boxes)
        
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.LongTensor(labels)
        target['image_id'] = torch.tensor([index])
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['iscrowd'] = torch.BoolTensor(iscrowd)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


def get_transforms(is_train=False):
    transforms = [ToTensor()]
    if is_train:
        transforms.append(RandomHorizontalFlip(0.5))
    
    return Compose(transforms)


def load_data(root, batch_size, train_split='train', val_split='val', class2ind_file='face_semantic_class2ind', use_objects=True, use_parts=False,
              return_separate_targets=False, part_class2ind_file=None, num_workers=0, max_samples=None):
    """
    `load train/val data loaders and class2ind (dict), n_classes (int)`

    Args:
        root (string): Root directory of the Pascal Part Dataset. Must contain the fololowing dir structure:
            Images: `root`/JPEGImages/*.jpg
            Object and Part annotations: `root`/Annotations_Part_json/*.json [see `parse_Pascal_VOC_Part_Anno.py`]
            train/val splits: `root`/ImageSets/Main/`image_set`.txt
            class2ind_file: `root`/Classes/`class2ind_file`.txt
        batch_size: batch size for training
        train/val splits: `root`/ImageSets/Main/`image_set`.txt
        class2ind_file: file containing list of class names and class index that are to be considered from all annotations.
            Other object/part classes will be ignored.
            Default: `face_semantic_class2ind`.
            Note: `__background__` class should also be present.
        use_objects: if True (default=True), use object annotations
        use_parts: if True (default=False), use part annotations that are present inside an object
        return_separate_targets: if True, return img, obj_target, part_target instead of img, target (default: False)
            should be set True only for training JointDetector
        part_class2ind_file: similar to `class2ind_file` but will have part classes (default: None).
            should be provided only if return_separate_targets=True otherwise should be provided as `class2ind_file`.
        max_samples: maximum number of samples for train/val datasets. (Default: None)
            Can be set to a small number for faster training
    """
    train_dataset = FaceDetection(root, train_split, get_transforms(is_train=True), class2ind_file=class2ind_file, use_objects=use_objects,
                                  use_parts=use_parts, return_separate_targets=return_separate_targets, part_class2ind_file=part_class2ind_file)
    val_dataset = FaceDetection(root, val_split, get_transforms(is_train=False), class2ind_file=class2ind_file, use_objects=use_objects,
                                use_parts=use_parts, return_separate_targets=return_separate_targets, part_class2ind_file=part_class2ind_file)

    class2ind = train_dataset.class2ind
    n_classes = train_dataset.n_classes

    if max_samples is not None:
        train_dataset = data.Subset(train_dataset, np.arange(max_samples))
        val_dataset = data.Subset(val_dataset, np.arange(max_samples))

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn,
                                   drop_last=True)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    print('Number of Samples --> Train:%d\t Val:%d\t' % (len(train_dataset), len(val_dataset)))

    return train_loader, val_loader, class2ind, n_classes