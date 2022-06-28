import torch
from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule
from torchvision import transforms
import torchvision.datasets as datasets
import os

import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id=0, 
                         num_shards=1, dali_cpu=False, is_training=True):
  
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels

  
class ImagenetteDaliDataModule(LightningDataModule):
    """
    
    A dataloader class for imagenette
    
    """
  
    name = 'imagenette'
    
    def __init__(self,
                data_dir: str,
                batch_size: int = 32,
                num_workers: int = 4
                ):

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.workers = num_workers
        self.num_classes = 10
        self.image_shape = [3, 224, 224]
        self.crop_size = 224
        self.val_size = 256
        
        # we need to work out these
        self.local_rank = 0
        self.global_rank = 0

    def setup(self, stage=None):
        
        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size
        
        imagenette_pipeline_train = create_dali_pipeline(batch_size=self.batch_size,
                                                         num_threads=self.workers,
                                                         device_id=device_id,
                                                         data_dir=os.path.join(self.data_dir, 'train'), 
                                                         crop=self.crop_size, 
                                                         size=self.val_size,
                                                         dali_cpu=False,
                                                         shard_id=shard_id, 
                                                         num_shards=num_shards,  
                                                         is_training=True)
        
        imagnette_pipeline_val = create_dali_pipeline(batch_size=self.batch_size,
                                                         num_threads=self.workers,
                                                         device_id=device_id,
                                                         data_dir=os.path.join(self.data_dir, 'val'), 
                                                         crop=self.crop_size, 
                                                         size=self.val_size,
                                                         dali_cpu=False,
                                                         shard_id=shard_id, 
                                                         num_shards=num_shards,  
                                                         is_training=False)
            
        class LightningWrapper(DALIClassificationIterator):
            def __init__(self, *kargs, **kvargs):
                super().__init__(*kargs, **kvargs)

            def __next__(self):
                out = super().__next__()
                # DDP is used so only one pipeline per process
                # also we need to transform dict returned by DALIClassificationIterator to iterable
                # and squeeze the lables
                out = out[0]
                return [out[k] if k != "label" else torch.squeeze(out[k]).long() for k in self.output_map]
            
        self.train_loader = LightningWrapper(imagenette_pipeline_train, reader_name="Reader",
                                                       last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
        
        self.val_loader = LightningWrapper(imagnette_pipeline_val, reader_name="Reader",
                                                       last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
        
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    