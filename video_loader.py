from __future__ import print_function, division
import numpy as np, os, time, cv2, pickle, json, torchvision, network, numbers
from torch.utils.data import Dataset
import torch, torchvision, torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F

def _is_tensor_clip(clip):
    return torch.is_tensor(clip) and clip.ndimension() == 4

def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped


def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = [
            cv2.resize(img, size, interpolation=np_inter) for img in clip
        ]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.NEAREST
        else:
            pil_inter = PIL.Image.BILINEAR
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def normalize(clip, mean, std, inplace=False):
    if not _is_tensor_clip(clip):
        raise TypeError('tensor is not a torch clip.')

    if not inplace:
        clip = clip.clone()

    dtype = clip.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

    return clip

def reshape(video, width, height): # transform video into standard sizing
    return np.asarray(resize_clip(video, (width, height), interpolation='nearest'))

def norm(video): # returns RGB scaled to a 0-1 scale
    return np.divide(video, np.full_like(video, 255))

def get_mapping(path, filename): # creates and serializes object for mapping
    # send dictonary to filename.p
    obj = {}
    i = 0
    for label in os.listdir(path):
        for f in os.listdir(path + "/" + label):
            if '.mp4' in f:
                obj[i] = [label + "/" + f, label]
                i += 1
    pickle.dump( obj, open( filename, "wb" ) )
    print("Pickled object!")

def get_labels(path): # get labels for network
    labels = {}
    i = 0
    for directory in os.listdir(path):
        labels[directory] = i
        i += 1
    return labels

def random_temporal_crop(video, length): # from random int if longer than specified length
    cropped_video = np.asarray(video)
    diff = len(video) - length
    if diff > 0:
        idx = np.random.randint(0, diff)
        return np.asarray(video[idx:idx + length])
    elif diff == 0:
        return np.asarray(video)
    else: 
        while (diff != 0):
            if diff < 0:
                cropped_video = np.append(cropped_video, np.asarray(video), axis=0)
                diff = len(cropped_video) - length
                if diff >= 0:
                    return cropped_video[:length]
                
def temporal_crop(video, length): # non-random from start crop
    cropped_video = np.asarray(video)
    diff = len(video) - length
    if diff >= 0:
        return np.asarray(video[:length])
    else: 
        while (diff != 0):
            if diff < 0:
                cropped_video = np.append(cropped_video, np.asarray(video), axis=0)
                diff = len(cropped_video) - length
                if diff >= 0:
                    return cropped_video[:length]
                
def center_spatial_crop(video, cropx, cropy):
    y,x = video.shape[1], video.shape[2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return video[:, starty:starty+cropy, startx:startx+cropx, :]

def random_spatial_crop(video, new_size):
    temp_h = video.shape[1] - new_size
    temp_w = video.shape[2] - new_size
    top = 0
    if (temp_h != 0):
        top = np.random.randint(0, temp_h)
    left = 0
    if (temp_w != 0):
        left = np.random.randint(0, temp_w)
    return video[:,top : top + new_size, left : left + new_size,:]

class GeneralVideoDataset(Dataset):
    """
    Dataset Class for Loading Video
    Args:
        clips_list_file (string): Path to the clipsList file with labels.
        root_dir (string): Directory with all the videos.
        channels: Number of channels of frames
        time_depth: Number of frames to be loaded in a sample
    """
    def __init__(self, clips_list_file, root_dir, replacement, replacement_label, channels, time_depth, x_size, y_size, test=False):
        
        with open(clips_list_file, "rb") as fp:  # Unpickling
            clips_list_file = pickle.load(fp)
        print("Dataset size: {}".format(len(clips_list_file)))
        self.clips_list = clips_list_file
        self.root_dir = root_dir
        self.channels = channels
        self.time_depth = time_depth
        self.x_size = x_size
        self.y_size = y_size
        self.test = test
        self.replacement = replacement
        self.replacement_label = replacement_label

    def __len__(self):
        return len(self.clips_list)

    def read_video(self, video_file):
        # Open the video file
        cap = cv2.VideoCapture(video_file)
        frames = []
        success, image = cap.read()
        while success:
            frames.append(image)
            success, image = cap.read()
        if len(frames) <= 1:
            return frames
        elif not self.test:
            return norm(random_temporal_crop(random_spatial_crop(np.asarray(frames), self.x_size), self.time_depth))
        else:
            return norm(temporal_crop(center_spatial_crop(np.asarray(frames), self.x_size, self.y_size), self.time_depth))
    
    def __getitem__(self, idx):
        video_file = os.path.join(self.root_dir, self.clips_list[idx][0])
        clip = self.read_video(video_file)
        if len(clip) <= 1:
            print("Got replacement clip!")
            sample = self.read_video(self.replacement)
            label = get_labels(self.root_dir)[self.replacement_label]
            return torch.from_numpy(sample), torch.from_numpy(np.asarray(label))
        else:
            sample = clip
            label = get_labels(self.root_dir)[self.clips_list[idx][1]]
            return torch.from_numpy(sample), torch.from_numpy(np.asarray(label))
    