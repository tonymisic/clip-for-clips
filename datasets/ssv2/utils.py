import os
import sys
import json
import pickle
import argparse
import torch
import shutil
import glob
import math
import numpy as np


def load_args():
    parser = argparse.ArgumentParser(description='2D Clip training ssv2')
    parser.add_argument('--config', '-c', default='datasets/ssv2/configs/config_supervised_CLIP.json', help='json config file path')
    # parser.add_argument('--config', '-c', default='datasets/ssv2/configs/config_3D_ResNet.json', help='json config file path')
    parser.add_argument('--clip', default=True,
                        help="train clip contrastive learning model.")
    parser.add_argument('--use_objects', default=True,
                        help="use the placeholder object instead of `something`.")
    parser.add_argument('--visual_clip_resume', default='3d3N50_CLIP.pth', action='store_true', help="resume visual encoder CLIP training from a given checkpoint.")
    parser.add_argument('--eval_only', '-e', action='store_true', #default=True,
                        help="evaluate trained model on validation data.")
    parser.add_argument('--resume', '-r', action='store_true',
                        help="resume training from a given checkpoint.")
    parser.add_argument('--gpus', '-g', default='0,1', help="GPU ids to use. Please"
                         " enter a comma separated list")
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help="to use GPUs")
    args = parser.parse_args()
    # if len(sys.argv) < 2:
    #     parser.print_help()
    #     sys.exit(1)
    return args


def remove_module_from_checkpoint_state_dict(state_dict):
    """
    Removes the prefix `module` from weight names that gets added by
    torch.nn.DataParallel()
    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def load_json_config(path):
    """ loads a json config file"""
    with open(path) as data_file:
        config = json.load(data_file)
        config = config_init(config)
    return config


def config_init(config):
    """ Some of the variables that should exist and contain default values """
    if "augmentation_mappings_json" not in config:
        config["augmentation_mappings_json"] = None
    if "augmentation_types_todo" not in config:
        config["augmentation_types_todo"] = None
    return config


def setup_cuda_devices(args):
    device_ids = []
    device = torch.device("cuda" if args.use_cuda else "cpu")
    if device.type == "cuda":
        device_ids = [int(i) for i in args.gpus.split(',')]
    return device, device_ids


def clip_loss(video_feature_batch, text_feature_batch, batch_size, CE_loss, labels, temperature=0.07): # temperature is learned ?
    # l2 normalization - THEY LINEARLY PROJECT HERE, WE (DO) NOT!
    video_feature_batch = video_feature_batch / video_feature_batch.norm(dim=-1, keepdim=True)
    text_feature_batch = text_feature_batch / text_feature_batch.norm(dim=-1, keepdim=True)

    # cosine sim as logits
    logit_scale = math.exp(temperature)
    logits = logit_scale * video_feature_batch @ text_feature_batch.t()
    # logits_per_text = logit_scale * text_feature_batch @ video_feature_batch.t()
    # generate labels
    # calculate loss of mini-batch
    video_loss = CE_loss(logits, labels)
    text_loss = CE_loss(logits.T, labels)
    overall_loss = (video_loss + text_loss) / 2
    return video_loss, text_loss, overall_loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def save_checkpoint(state, is_best, config, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(config['output_dir'], config['model_name'], filename)
    model_path = os.path.join(config['output_dir'], config['model_name'], 'model_best.pth.tar')
    torch.save(state, checkpoint_path)
    if is_best:
        print(" > Best model found at this epoch. Saving ...")
        shutil.copyfile(checkpoint_path, model_path)


def save_results(logits_matrix, features_matrix, targets_list, item_id_list,
                 class_to_idx, config):
    """
    Saves the predicted logits matrix, true labels, sample ids and class
    dictionary for further analysis of results
    """
    print("Saving inference results ...")
    path_to_save = os.path.join(
        config['output_dir'], config['model_name'], "test_results.pkl")
    with open(path_to_save, "wb") as f:
        pickle.dump([logits_matrix, features_matrix, targets_list,
                     item_id_list, class_to_idx], f)


def save_images_for_debug(dir_img, imgs):
    """
    2x3x12x224x224 --> [BS, C, seq_len, H, W]
    """
    print("Saving images to {}".format(dir_img))
    from matplotlib import pylab as plt
    imgs = imgs.permute(0, 2, 3, 4, 1)  # [BS, seq_len, H, W, C]
    imgs = imgs.mul(255).numpy()
    if not os.path.exists(dir_img):
        os.makedirs(dir_img)
    print(imgs.shape)
    for batch_id, batch in enumerate(imgs):
        batch_dir = os.path.join(dir_img, "batch{}".format(batch_id + 1))
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
        for j, img in enumerate(batch):
            plt.imsave(os.path.join(batch_dir, "frame{%04d}.png" % (j + 1)),
                       img.astype("uint8"))


def get_submission(logits_matrix, item_id_list, class_to_idx, config):
    top5_classes_pred_list = []

    for i, id in enumerate(item_id_list):
        logits_sample = logits_matrix[i]
        logits_sample_top5  = logits_sample.argsort()[-5:][::-1]
        # top1_class_index = logits_sample.argmax()
        # top1_class_label = class_to_idx[top1_class_index]

        top5_classes_pred_list.append(logits_sample_top5)

    path_to_save = os.path.join(
            config['output_dir'], config['model_name'], "test_submission.csv")
    with open(path_to_save, 'w') as fw:
        for id, top5_pred in zip(item_id_list, top5_classes_pred_list):
            fw.write("{}".format(id))
            for elem in top5_pred:
                fw.write(";{}".format(elem))
            fw.write("\n")


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ExperimentalRunCleaner(object):
    """
    Remove the output dir, if you exit with Ctrl+C and if there are less
    then 1 file. It prevents the noise of experimental runs.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir

    def __call__(self, signal, frame):
        num_files = len(glob.glob(self.save_dir + "/*"))
        if num_files < 1:
            print('Removing: {}'.format(self.save_dir))
            shutil.rmtree(self.save_dir)
        print('You pressed Ctrl+C!')
        sys.exit(0)
