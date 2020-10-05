"""Classes to save checkponits and image during training and validation.

"""
import torch
import numpy as np
import nibabel as nib
from collections import namedtuple
from pathlib import Path
from queue import Queue
from threading import Thread
from enum import Enum

from .observer import Observer
from .utils import NamedData


class Saver(Observer):
    """An abstract class to save the training progress.

    Args:
        dirname (str or pathlib.Path): The directory to save results.
        save_init (bool): Save before any weight update.

    """
    def __init__(self, dirname, save_init=False):
        super().__init__()
        self.dirname = Path(dirname)
        self.save_init = save_init

    def update_on_train_start(self):
        """Creates the results folder."""
        if self.save_init:
            self._save()

    def _save(self):
        """Implement save in this function."""
        raise NotImplementedError


class ThreadedSaver(Saver):
    """Saves with threads.

    """
    def __init__(self, dirname, save_init=False):
        super().__init__(dirname, save_init)
        self._thread = self._init_thread()

    def _init_thread(self):
        raise NotImplementedError

    def update_on_train_start(self):
        super().update_on_train_start()
        self._thread.start()
    
    def update_on_train_end(self):
        self._thread.join()


class CheckpointSaver(Saver):
    """Saves model periodically.

    Attributes:
        step (int): Save a checkpoint every this number of epochs.
        kwargs (dict): The other stuff to save.

    """
    def __init__(self, dirname, step=100, save_init=False, **kwargs):
        super().__init__(dirname, save_init)
        self.step = step
        self.kwargs = kwargs

    def update_on_train_start(self):
        self.dirname.mkdir(parents=True, exist_ok=True)
        pattern = 'epoch-%%0%dd.pt' % len(str(self.subject.num_epochs))
        self._pattern = str(self.dirname.joinpath(pattern))
        super().update_on_train_start()

    def update_on_epoch_end(self):
        """Saves a checkpoint."""
        if self.subject.epoch_ind % self.step == 0:
            self._save()

    def _save(self):
        filename = self._pattern % self.subject.epoch_ind
        contents = {'epoch': self.subject.epoch_ind,
                    'model_state_dict': self.subject.get_model_state_dict(),
                    'optim_state_dict': self.subject.get_optim_state_dict(),
                    **self.kwargs}
        torch.save(contents, filename)


class SaveType(str, Enum):
    """The type of :class:`SaveImage`.

    Attributes:
        NIFTI: Save the image as a nifit file.

    """
    NIFTI = 'nifti'


class ImageType(str, Enum):
    """The type of image to save.

    Attributes:
        IMAGE: Just an image.
        SEG: The image is a segmentation.
        SEG_ACTIV: Apply activation (sigmoid) to the segmentation.
    
    """
    IMAEG = 'image'
    SEG = 'seg'
    SEG_ACTIV = 'seg_activ'


def create_save_image(save_type, image_type):
    """Creates an instance of :class:`SaveImage`.

    Args:
        save_type (enum SaveType or str): The type of :class:`SaveImage`.
        image_type (enum ImageType or str): The type of image to save.

    Returns:
        SaveImage: An instance of :class:`SaveImage`.
        
    """
    save_type = SaveType(save_type)
    image_type = ImageType(image_type)
    if save_type is SaveType.NIFTI:
        save_image = SaveNifti()
    if image_type is ImageType.SEG:
        save_image = SaveSeg(save_image)
    elif image_type is ImageType.SEG_ACTIV:
        save_image = SaveSegActiv(save_image)
    return save_image


class SaveImage:
    """Writes images to dick.
    
    """
    def save(self, filename, image):
        """Saves an image to filename.

        Args:
            filename (str or pathlib.Path): The filename to save.
            image (numpy.ndarray): The image to save.
        
        """
        raise NotImeplementedError


class SaveNifti(SaveImage):
    """Writes images as nifti files.
    
    """
    def save(self, filename, image):
        filename = str(filename)
        if not filename.endswith('.nii') and not filename.endswith('.nii.gz'):
            filename = filename + '.nii.gz'
        obj = nib.Nifti1Image(image.numpy(), np.eye(4))
        obj.to_filename(filename)


class SaveSeg(SaveImage):
    """Saves a segmentation to file.

    Attributes:
        save_image (SaveImage): The instance to wrap around.
    
    """
    def __init__(self, save_image):
        self.save_image = save_image

    def save(self, filename, image):
        image = self._convert_seg(image)
        self.save_image.save(filename, image)

    def _convert_seg(self, image):
        """Converts a probability map to segmentation."""
        if image.shape[0] > 1:
            image = torch.argmax(image, dim=0, keepdim=True)
        return image 


class SaveSegActiv(SaveSeg):
    """Applies activation before saving the segmentation.
    
    """
    def _convert_seg(self, image):
        if image.shape[0] > 1:
            image = torch.argmax(image, dim=0, keepdim=True)
        else:
            image = torch.sigmoid(image)
        return image


class ImageThread(Thread):
    """Saves images in a thread.

    Attributes:
        save_image (SaveImage): Save images to files.
    
    """
    def __init__(self, save_image, queue):
        super().__init__()
        self.save_image = save_image
        self.queue = queue

    def run(self):
        while True:
            data = self.queue.get()
            self.queue.task_done()
            if data is None:
                break
            self.save_image.save(data.name, data.data)


class ImageSaver(ThreadedSaver):
    """Saves images.

    Attributes:
        attrs (list[str]): The attribute names of :attr:`subject` to save.
        step (int): Save images every this number of epochs.
        queue (queue.Queue): The queue to give data to its thread.
        save_type (str): The type of files to save the images to.
        image_type (str): The type of images to save.
        file_struct (str): Indicates the file structure. For example,
            ``"epoch/batch/sample"`` saves all samples of a mini-batch as
            `dirname/epoch-ind/batch-ind/sample-ind_name.ext``.
            ``"epoch/batch"`` saves the samples as
            ``dirname/epoch-ind/batch-ind_sample-ind_name.ext``.
    
    """
    def __init__(self, dirname, attrs=[], step=10, save_type='nifti',
                 image_type='image', file_struct='epoch/batch/sample'):
        self.save_type = save_type
        self.image_type = image_type
        self.queue = Queue()
        self.attrs = attrs
        self.step = step
        self.file_struct = file_struct
        self._pattern = None
        super().__init__(dirname)

    def update_on_train_start(self):
        self._pattern = self._get_filename_pattern()
        super().update_on_train_start()

    def _init_thread(self):
        save_image = create_save_image(self.save_type, self.image_type)
        return ImageThread(save_image, self.queue)

    def _get_filename_pattern(self):
        epoch_pattern = 'epoch-%%0%dd' % len(str(self.subject.num_epochs))
        batch_pattern = 'batch-%%0%dd' % len(str(self.subject.num_batches))
        sample_pattern = 'sample-%%0%dd' % len(str(self.subject.batch_size))
        all_parts = ['epoch', 'batch', 'sample']
        file_parts = self.file_struct.split('/')
        assert set(file_parts).issubset(set(all_parts))
        pattern = self.dirname
        for ap in all_parts:
            sub_pattern = eval('_'.join([ap, 'pattern']))
            pattern = self._join_pattern(sub_pattern, pattern, ap in file_parts)
        pattern = self._join_pattern('%s', pattern, False)
        return pattern

    def _join_pattern(self, sub, all, as_path=True):
        """Concatenates ``sub_pattern`` after ``pattern``."""
        return str(Path(all, sub)) if as_path else '_'.join([all, sub])

    def update_on_batch_end(self):
        if self.subject.epoch_ind % self.step == 0:
            self._save()

    def _save(self):
        for attr in self.attrs:
            batch = getattr(self.subject, attr)
            if isinstance(batch, NamedData):
                for sample_ind, (name, sample) in enumerate(zip(*batch)):
                    filename = self._get_filename(sample_ind + 1, attr)
                    filename = '_'.join([filename, name])
                    self.queue.put(NamedData(filename, sample))
            else:
                for sample_ind, sample in enumerate(batch):
                    filename = self._get_filename(sample_ind + 1, attr)
                    self.queue.put(NamedData(filename, sample))

    def _get_filename(self, sample_ind, name):
        filename = self._pattern % (self.subject.epoch_ind,
                                    self.subject.batch_ind,
                                    sample_ind, name)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        return filename

    def update_on_train_end(self):
        self.queue.put(None)
        super().update_on_train_end()
