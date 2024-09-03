# helper function for data visualization
import os
import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name).title())
        plt.imshow(image)
    plt.show()
    
# helper function for data visualization
def save(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(6.72, 6.72))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, 1, 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)
        plt.savefig(name+str(i))
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


# classes for data loading and preprocessing
class DatasetView:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['green', 'no_green']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['green', 'no_green']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.images_fps = [os.path.join(images_dir, fname) for fname in os.listdir(images_dir)]
        self.masks_fps = [os.path.join(masks_dir, fname) for fname in os.listdir(masks_dir)]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def get_images_and_masks(self):
        images = []
        masks = []
        
        for img_fp, mask_fp in zip(self.images_fps, self.masks_fps):
            # read data
            image = cv2.imread(img_fp)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE)
            
            # extract certain classes from mask
            masks_array = [(mask == v) for v in self.class_values]
            mask = np.stack(masks_array, axis=-1).astype('float')
            
            # add background if mask is not binary
            if mask.shape[-1] != 1:
                background = 1 - mask.sum(axis=-1, keepdims=True)
                mask = np.concatenate((mask, background), axis=-1)
            
            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            
            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
                
            images.append(image)
            masks.append(mask)
        
        return images, masks
    
    def to_tf_dataset(self, batch_size=32):
        def generator():
            images, masks = self.get_images_and_masks()
            for img, mask in zip(images, masks):
                yield img, mask

        dataset = tf.data.Dataset.from_generator(generator,
                                                 output_signature=(
                                                     tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                                                     tf.TensorSpec(shape=(None, None, len(self.CLASSES) + 1), dtype=tf.float32)
                                                 ))
        dataset = dataset.batch(batch_size)
        return dataset
    
    def __len__(self):
        return len(self.images_fps)
    
    
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=32, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def __getitem__(self, index):
        start = index * self.batch_size
        stop = (index + 1) * self.batch_size
        batch_data = [self.dataset[i] for i in range(start, stop)]

        # Unzip batch data
        images, masks = zip(*batch_data)

        # Convert to numpy arrays and stack
        X_batch = np.stack(images, axis=0)
        y_batch = np.stack(masks, axis=0)

        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)