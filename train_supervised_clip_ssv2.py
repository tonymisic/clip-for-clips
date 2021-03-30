import os
import sys
import time
import signal
import importlib
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import clip, network
# import video_loader as vl, pickle
from PIL import Image
import matplotlib.pyplot as plt


from datasets.ssv2.utils import *
from datasets.ssv2.callbacks import (PlotLearning, AverageMeter)
# from datasets.ssv2.models.multi_column import MultiColumn
# import torchvision
from torch.utils.tensorboard import SummaryWriter
from datasets.ssv2.transforms_video import *
# import threading

from torchvideotransforms import video_transforms, volume_transforms


cv2.setNumThreads(0)

print(torch.version.cuda)
# load configurations
args = load_args()
config = load_json_config(args.config)

# set column model
file_name = config['conv_model']
cnn_def = importlib.import_module("{}".format(file_name))

# setup device - CPU or GPU
device, device_ids = setup_cuda_devices(args)
print(" > Using device: {}".format(device.type))
print(" > Active GPU ids: {}".format(device_ids))

best_loss = float('Inf')
best_acc = 0

if config["input_mode"] == "av":
    from datasets.ssv2.data_loader_av import VideoFolder
elif config["input_mode"] == "skvideo":
    from datasets.ssv2.data_loader_skvideo import VideoFolder
else:
    raise ValueError("Please provide a valid input mode")


def main():
    global args, best_loss, start_epoch, best_acc

    # set run output folder
    model_name = config["model_name"]
    output_dir = config["output_dir"]
    save_dir = os.path.join(output_dir, model_name)
    print(" > Output folder for this run -- {}".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'plots'))

    if not args.eval_only:
        writer = SummaryWriter(save_dir)

    # assign Ctrl+C signal handler
    signal.signal(signal.SIGINT, ExperimentalRunCleaner(save_dir))

    # create model
    print(" > Creating model ... !")
    if args.clip:
        res3d = network.generate_model(50)
        model, preprocess = clip.load("RN50", device='cpu')
        model.visual = res3d
        if os.path.exists(save_dir + '/model_best.pth.tar'):
            print(' > Loading best checkpoint from:', save_dir + '/model_best.pth.tar')
            checkpoint = torch.load(save_dir + '/model_best.pth.tar')
            best_acc = checkpoint['val_top1']
            start_epoch = checkpoint['epoch']
            visual_chkpt = checkpoint['visul_state_dict']
            text_chkpt = checkpoint['text_state_dict']
            model.visual.load_state_dict(visual_chkpt)
            model.transformer.load_state_dict(text_chkpt)

        elif args.visual_clip_resume:
            print('loading inflated weights!')
            model.visual.load_state_dict(torch.load(args.visual_clip_resume), strict=False)

        if args.resume:
            checkpoint = torch.load(save_dir + '/model_best.pth')
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
        ### NEED THE FULL CLIP MODEL (i.e., CLIP.visual) by this point ###
    else:
        # model = MultiColumn(config['num_classes'], cnn_def.Model, int(config["column_units"]))
        model = network.generate_model(50)
        # model.fc = nn.Linear(2048, 1024)
        if os.path.exists(save_dir + '/model_best.pth.tar'):
            print(' > Loading best checkpoint from:', save_dir + '/model_best.pth.tar')
            model.fc = nn.Linear(2048, 174)
            checkpoint = torch.load(save_dir + '/model_best.pth.tar')
            best_acc = checkpoint['val_top1']
            best_loss = checkpoint['best_loss']
            start_epoch = checkpoint['epoch']
            chkpt = checkpoint['state_dict']
            model.load_state_dict(chkpt, strict=False)
        elif args.visual_clip_resume:
            chkpt = torch.load(args.visual_clip_resume)
            print('loading inflated weights!')
            # model.load_state_dict(chkpt)
            model.fc = nn.Linear(2048, 174)
            # model.fc = nn.Linear(2048, 174)


    # multi GPU setting

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        # model.visual = nn.DataParallel(model.visual)
        # model.transformer = nn.DataParallel(model.transformer)
        model = nn.DataParallel(model)


    model.to(device)


    # model = torch.nn.DataParallel(model, device_ids).to(device)
    # model = model.type(torch.FloatTensor).to(device)

    # optionally resume from a checkpoint
    # checkpoint_path = os.path.join(config['output_dir'],
    #                                config['model_name'],
    #                                'model_best.pth.tar')
    # if args.resume:
    #     if os.path.isfile(checkpoint_path):
    #         print(" > Loading checkpoint '{}'".format(args.resume))
    #         # if args.clip:
    #         #     res3d = network.generate_model(50)
    #         #     model, preprocess = clip.load("RN50", device=device)
    #         #     model.visual = res3d
    #         #     if args.visual_clip_resume:
    #         #         model.visual.load_state_dict(torch.load(checkpoint_path))
    #         checkpoint = torch.load(checkpoint_path)
    #         args.start_epoch = checkpoint['epoch']
    #         best_loss = checkpoint['best_loss']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         print(" > Loaded checkpoint '{}' (epoch {})"
    #               .format(checkpoint_path, checkpoint['epoch']))
    #     else:
    #         print(" !#! No checkpoint found at '{}'".format(
    #             checkpoint_path))

    # define augmentation pipeline
    upscale_size_train = int(config['input_spatial_size'] * config["upscale_factor_train"])
    upscale_size_eval = int(config['input_spatial_size'] * config["upscale_factor_eval"])

    # OPENCV TRANSFORMS
    # Random crop videos during training
    # transform_train_pre = ComposeMix([
    #         [RandomRotationVideo(15), "vid"],
    #         [Scale(upscale_size_train), "img"],
    #         [RandomCropVideo(config['input_spatial_size']), "vid"],
    #          ])

    # PYTORCH TRANSFORMS
    transform_train_pre = video_transforms.Compose([
            video_transforms.RandomRotation(15),
            video_transforms.Resize(upscale_size_train),
            video_transforms.RandomCrop((config['input_spatial_size'],config['input_spatial_size'])),
            volume_transforms.ClipToTensor()
             ])

    # Center crop videos during evaluation
    # transform_eval_pre = ComposeMix([
    #         [Scale(upscale_size_eval), "img"],
    #         [torchvision.transforms.ToPILImage(), "img"],
    #         [torchvision.transforms.CenterCrop(config['input_spatial_size']), "img"],
    #          ])

    # Center crop videos during evaluation
    transform_eval_pre = video_transforms.Compose([
            video_transforms.Resize(upscale_size_eval),
            video_transforms.CenterCrop((config['input_spatial_size'],config['input_spatial_size'])),
            volume_transforms.ClipToTensor()
             ])
    # transform_post = ComposeMix([
    #         [torchvision.transforms.ToTensor(), "img"],
    #         [torchvision.transforms.Normalize(
    #                    mean=[0.485, 0.456, 0.406],  # default values for imagenet
    #                    std=[0.229, 0.224, 0.225]), "img"]
    #          ])
    # Transforms common to train and eval sets and applied after "pre" transforms
    # transform_post = ComposeMix([
    #         [torchvision.transforms.ToTensor(), "img"],
    #         [torchvision.transforms.Normalize(
    #                    mean=[0.485, 0.456, 0.406],  # default values for imagenet
    #                    std=[0.229, 0.224, 0.225]), "img"]
    #          ])

    train_data = VideoFolder(root=config['data_folder'],
                             json_file_input=config['json_data_train'],
                             json_file_labels=config['json_file_labels'],
                             clip_size=config['clip_size'],
                             nclips=config['nclips_train'],
                             step_size=config['step_size_train'],
                             is_val=False,
                             transform_pre=transform_train_pre,
                             transform_post=None,
                             augmentation_mappings_json=config['augmentation_mappings_json'],
                             augmentation_types_todo=config['augmentation_types_todo'],
                             get_item_id=False,
                             use_objects=args.use_objects
                             )

    print(" > Using {} processes for data loader.".format(
        config["num_workers"]))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=True)

    val_data = VideoFolder(root=config['data_folder'],
                           json_file_input=config['json_data_val'],
                           json_file_labels=config['json_file_labels'],
                           clip_size=config['clip_size'],
                           nclips=config['nclips_val'],
                           step_size=config['step_size_val'],
                           is_val=True,
                           transform_pre=transform_eval_pre,
                           transform_post=None,
                           get_item_id=True,
                           use_objects=args.use_objects
                           )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=False)

    # test_data = VideoFolder(root=config['data_folder'],
    #                         json_file_input=config['json_data_test'],
    #                         json_file_labels=config['json_file_labels'],
    #                         clip_size=config['clip_size'],
    #                         nclips=config['nclips_val'],
    #                         step_size=config['step_size_val'],
    #                         is_val=True,
    #                         transform_pre=transform_eval_pre,
    #                         transform_post=transform_post,
    #                         get_item_id=True,
    #                         is_test=True,
    #                         use_objects=args.use_objects
    #                         )

    # test_loader = torch.utils.data.DataLoader(
    #     test_data,
    #     batch_size=config['batch_size'], shuffle=False,
    #     num_workers=config['num_workers'], pin_memory=True,
    #     drop_last=False)

    print(" > Number of dataset classes : {}".format(len(train_data.classes)))
    assert len(train_data.classes) == config["num_classes"]

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    lr = config["lr"]
    last_lr = config["last_lr"]
    momentum = config['momentum']
    weight_decay = config['weight_decay']

    for param in model.module.transformer.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(model.module.visual.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    if args.eval_only:
        validate(val_loader, model, criterion, train_data.classes_dict)
        print(" > Evaluation DONE !")
        return

    # set callbacks
    # plotter = PlotLearning(os.path.join(
    #     save_dir, "plots"), config["num_classes"])
    lr_decayer = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, 'min', factor=config["lr_factor"], patience=config["lr_patience"], verbose=True)
    # val_loss = float('Inf')

    # set end condition by num epochs
    num_epochs = int(config["num_epochs"])
    if num_epochs == -1:
        num_epochs = 999999

    print(" > Training is getting started...")
    print(" > Training takes {} epochs.".format(num_epochs))
    # start_epoch = args.start_epoch if args.resume else 0

    if best_acc > 0:
        print('Best Val Acc: ', best_acc)
        print('Best train loss: ', best_loss)
        print('Starting at epoch: ', start_epoch)
    else:
        start_epoch = args.start_epoch if args.resume else 0

    for epoch in range(start_epoch, num_epochs):
        lrs = [params['lr'] for params in optimizer.param_groups]
        print(" > Current LR(s) -- {}".format(lrs))
        if np.max(lr) < last_lr and last_lr > 0:
            print(" > Training is DONE by learning rate {}".format(last_lr))
            sys.exit(1)

        # train for one epoch
        train_loss = train(
            train_loader, model, criterion, optimizer, epoch, writer)

        writer.add_scalar('Ave_Loss/Ave_train_Loss', train_loss, epoch)

        # evaluate on validation set
        val_loss, val_top1, val_top5 = validate(val_loader, model, criterion)
        writer.add_scalar('Ave_Loss/Val_loss', val_loss, epoch)
        writer.add_scalar('Val_acc1', val_top1, epoch)
        writer.add_scalar('Val_acc5', val_top5, epoch)

        # set learning rate
        lr_decayer.step(train_loss)

        # plot learning
        # plotter_dict = {}
        # plotter_dict['loss'] = train_loss
        # plotter_dict['val_loss'] = val_loss
        # plotter_dict['acc'] = train_top1 / 100
        # plotter_dict['val_acc'] = val_top1 / 100
        # plotter_dict['learning_rate'] = lr
        # plotter.plot(plotter_dict)

        print(" > Train loss after epoch {} = {}".format(epoch, train_loss))

        # remember best loss and save the checkpoint
        is_best = train_loss < best_loss
        best_loss = min(train_loss, best_loss)

        if args.clip:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': "3DR_CLIP",
                'visul_state_dict': model.module.visual.state_dict(),
                'text_state_dict': model.module.transformer.state_dict(),
                'val_top1': val_top1,
                'val_top5': val_top5,
                'lrs': lrs,
                'best_loss': best_loss,
            }, is_best, config)
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': "3dresnet50",
                'state_dict': model.module.state_dict(),
                'val_top1': val_top1,
                'val_top5': val_top5,
                'lrs': lrs,
                'best_loss': best_loss,
            }, is_best, config)


def train(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    CE_losses = AverageMeter()
    clip_losses = AverageMeter()
    batch_size = config['batch_size']
    labels = torch.arange(batch_size).type(torch.LongTensor).to(device)
    # switch to train mode
    model.train()
    num_iters = len(train_loader)
    end = time.time()
    # for i, (input, target, _, _) in enumerate(train_loader):
    for i, (input, target, obj_caption, template_caption) in enumerate(train_loader):
    # for i, (input1, target1, obj_caption1, template_caption1) in enumerate(train_loader):

        # if i == 0:
        #     input = input1
        #     target = target1
        #     obj_caption = obj_caption1
        #     template_caption = template_caption1

        # measure data loading time
        data_time.update(time.time() - end)

        # if config['nclips_train'] > 1:
        #     input_var = list(input.split(config['clip_size'], 2))
        #     for idx, inp in enumerate(input_var):
        #         input_var[idx] = inp.to(device)
        # else:
        #     input_var = [input.to(device)]
        model.zero_grad()
        # move to cuda
        # input = input.type(torch.FloatTensor).to(device)
        input = input.to(device)

        if args.clip:
            target = target.type(torch.LongTensor).to(device)
            caption = clip.tokenize(template_caption).to(device)
            video_features, text_features, output = model(input, caption, pred=True)
            vid_loss, text_loss, CLP_loss = clip_loss(video_features, text_features, video_features.shape[0], criterion, labels)
            CE_Loss = criterion(output, target)
            loss = CE_Loss + CLP_loss
        else:
            target = target.to(device)
            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = accuracy(logits.detach().cpu(), target.detach().cpu(), topk=(1, 5))


        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.detach().cpu(), target.detach().cpu(), topk=(1, 5))
        losses.update(loss.detach().cpu().item(), input.size(0))
        CE_losses.update(CE_Loss.detach().cpu().item(), input.size(0))
        clip_losses.update(CLP_loss.detach().cpu().item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % config["print_freq"] == 0:
        # if i % 1 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'CE_Loss {CE_Loss.val:.4f} ({loss.avg:.4f})\t'
                  'CLIP_Loss {CLIP_Loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, CE_Loss=CE_losses, CLIP_Loss=clip_losses, top1=top1, top5=top5))
        # break
        # add to tensorboard
        if i % config["tensorboard_scalar_add"] == 0:
            writer.add_scalar('train_loss', loss.detach().cpu().item(), epoch*num_iters + i)
            writer.add_scalar('train_CE_loss', CE_Loss.detach().cpu().item(), epoch*num_iters + i)
            writer.add_scalar('train_CLIP_loss', CLP_loss.detach().cpu().item(), epoch*num_iters + i)
            writer.add_scalar('train_acc1', top1.val, epoch*num_iters + i)
            writer.add_scalar('train_acc5', top5.val, epoch*num_iters + i)

    return losses.avg


def validate(val_loader, model, criterion, class_to_idx=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    f = open("datasets/ssv2/classes.txt", "r")
    classes = []
    for line in f.readlines():
        classes.append(line.rstrip('\n'))
    text = clip.tokenize(classes).to(device)
    batch_size = config['batch_size']
    labels = torch.arange(batch_size).type(torch.LongTensor).to(device)
    # switch to evaluate mode
    model.eval()

    logits_matrix = []
    features_matrix = []
    targets_list = []
    item_id_list = []
    logit_scale = nn.Parameter(torch.ones([]))
    end = time.time()
    with torch.no_grad():
        # for i, (input, target, item_id) in enumerate(val_loader):
        for i, (input, target, item_id, obj_caption, template_caption) in enumerate(val_loader):

            # if config['nclips_val'] > 1:
            #     input_var = list(input.split(config['clip_size'], 2))
            #     for idx, inp in enumerate(input_var):
            #         input_var[idx] = inp.to(device)
            # else:
            #     input_var = [input.to(device)]

            input = input.to(device)
            target = target.to(device)
            if args.clip:
                caption = clip.tokenize(template_caption).to(device)
                _, _, logits = model(input, caption, pred=True)


                # logits_per_image = logit_scale * video_features @ text_features.t()
                # logits_per_text = logit_scale * text_features @ video_features.t()
                # probs = logits_per_image.detach().softmax(dim=-1).cpu().numpy()
                # output = np.argmax(probs, axis=1)
                # logits = logits_per_image.softmax(dim=-1)
                # preds = logits.argmax(-1)
                # correct = preds == target
                # correct = correct.detach().cpu().sum() / 174

                # if video_features.shape[0] == batch_size:
                #     vid_loss, text_loss, loss = clip_loss(video_features, caption_features, video_features.shape[0], criterion, labels)

            loss = criterion(logits, target)
            preds = logits.argmax(-1)

            # compute output and loss
            # output, features = model(input_var, config['save_features'])
            # loss = criterion(output, target)

            if args.eval_only:
                logits_matrix.append(preds.cpu().data.numpy())
                # features_matrix.append(features.cpu().data.numpy())
                targets_list.append(target.cpu().numpy())
                item_id_list.append(item_id)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits.detach().cpu(), target.detach().cpu(), topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config["print_freq"] == 0:
            # if i % 1 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5))


            # break

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    # if args.eval_only:
    #     logits_matrix = np.concatenate(logits_matrix)
    #     features_matrix = np.concatenate(features_matrix)
    #     targets_list = np.concatenate(targets_list)
    #     item_id_list = np.concatenate(item_id_list)
    #     print(logits_matrix.shape, targets_list.shape, item_id_list.shape)
    #     save_results(logits_matrix, features_matrix, targets_list,
    #                  item_id_list, class_to_idx, config)
    #     get_submission(logits_matrix, item_id_list, class_to_idx, config)
    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()

