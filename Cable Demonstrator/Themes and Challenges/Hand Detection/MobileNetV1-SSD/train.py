from __future__ import print_function
"""Python"""
import os
import time
import sys
import argparse
import logging
import logging.handlers
import torch
import config
import itertools
from utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from utils.data_preprocessing import TrainAugmentation, TestTransform
from models.mobilenetV1_SSD import MatchPrior
from models.mobilenetV1_SSD import MobileNetV1_SSD
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from data.voc.voc_dataset import VOCDataset
from data.egohands.egohands_dataset import EgoHandsDataset
from utils.multibox_loss import MultiboxLoss


"""
#######################
Description
#######################

This script focuses upon the training of a MobileNetV1 based implementation for single shot multibox object detection. 
Its target objects are the human hand in both left and right configurations, trained using a number of datasets, but 
with a focus on egocentric or overhead views. There are options to train on the original VOC2012 dataset as well.

To begin with you will need to download and setup the training data used. From the original repository there is the ability
to train on the VOC dataset using the 'data/voc' folder, download the datset to data/voc/VOC2012 (Annotations, ImageSets etc)
and then run the generate_vocdata.py script to correctly setup the training data.

To run use: python train_ssd.py --dataset_type voc --datasets ~/data/voc/VOC2012/ --validation_dataset ~/data/voc/VOC2012/ --keep_difficult True --base_net models/mobilenet-v1-ssd-mp-0_675.pth  --batch_size 24 --num_epochs 200 --scheduler cosine --lr 0.01 --t_max 200

However a pretrained model is already available and stored in 'models/mobilenet-v1-ssd-mp-0_675.pth', you can use this
as a base model instead when training on the EgoHands dataset. In order to do so you need to download the dataset for egohands
which can be found at: http://vision.soic.indiana.edu/egohands_files/egohands_data.zip (1.3GB Version), and arrange such
that you have a 'egohands/egohands_data/_LABELLED_SAMPLES' path.

Then you need to use this data and convert it to a VOC pascal format. This can be done by running the 'generate_egohandsVOCdata.py' script
This should give you a new group of folders 'egohandsVOC/ (Annotations, ImageSets, JPEGImages). You can now train on this using:

To run use: python train_ssd.py --dataset_type egohands --datasets ~/data/egohands/egohandsVOC/ --validation_dataset ~/data/egohands/egohandsVOC/ --keep_difficult True --base_net models/mobilenet-v1-ssd-mp-0_675.pth  --batch_size 24 --num_epochs 200 --scheduler cosine --lr 0.01 --t_max 200

For both examples i used keep_diffult as True, as it would crash on some of the sample images for some reason related to no bbox.
I also have set number of workers=0 as multithreading was causing locks / infitie loops for the dataloader on my old macbook :)

Based upon https://github.com/qfgaohao/pytorch-ssd/
"""


"""
#######################
Arguments
#######################
"""
def get_args():
    # Set parsed arguments
    parser = argparse.ArgumentParser(
        description='MobileNetV1 single shot multibox object detector')

    parser.add_argument("--dataset_type", default="voc", type=str,
                        help='Specify dataset type. Currently support voc and egohands.')
    # Params for datasets
    parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
    parser.add_argument('--validation_dataset', help='Dataset directory path')
    parser.add_argument('--keep_difficult', default=False, type=bool, help='Keep difficult images in datset')
    parser.add_argument('--balance_data', action='store_true',
                        help="Balance training data by down-sampling more frequent labels.")

    parser.add_argument('--freeze_base_net', action='store_true',
                        help="Freeze base net layers.")
    parser.add_argument('--freeze_net', action='store_true',
                        help="Freeze all the layers except the prediction head.")

    parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                        help='Width Multiplifier for MobilenetV2')

    # Params for SGD
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--base_net_lr', default=None, type=float,
                        help='initial learning rate for base net.')
    parser.add_argument('--extra_layers_lr', default=None, type=float,
                        help='initial learning rate for the layers not in base net and prediction heads.')

    # Params for loading pretrained basenet or checkpoints.
    parser.add_argument('--base_net',
                        help='Pretrained base model')
    parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')

    # Scheduler
    parser.add_argument('--scheduler', default="multi-step", type=str,
                        help="Scheduler for SGD. It can one of multi-step and cosine")

    # Params for Multi-step Scheduler
    parser.add_argument('--milestones', default="80,100", type=str,
                        help="milestones for MultiStepLR")

    # Params for Cosine Annealing
    parser.add_argument('--t_max', default=120, type=float,
                        help='T_max value for Cosine Annealing Scheduler.')

    # Train params
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', default=120, type=int,
                        help='the number epochs')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--validation_epochs', default=5, type=int,
                        help='the number epochs')
    parser.add_argument('--debug_steps', default=100, type=int,
                        help='Set the debug log output frequency.')
    parser.add_argument('--use_cuda', default=True, type=str2bool,
                        help='Use CUDA to train model')

    parser.add_argument('--checkpoint_folder', default='models/',
                        help='Directory for saving checkpoint models')

    return parser.parse_args()


"""
#######################
Train Model
#######################
"""
def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    # Set losses to zero
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    time.sleep(2)  # Prevent possible deadlock during epoch transition
    # Run through dataset and train network
    for i, data in enumerate(loader):
        # Start batch
        timer.start("Start batch")
        # From data get images, boxes and associated labels
        images, boxes, labels = data
        # Pass to device
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        # Set optimiser
        optimizer.zero_grad()
        # Pass to network for inference, and retrieve back label confidences and labels predicted for image/s
        confidence, locations = net(images)
        # Calculate loss
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()
        # Gather loss statistics
        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        # Check whether to log information
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            batch_time = timer.end("Start batch")
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}, " +
                f"Batch Run Time (s): {batch_time:.2f} seconds"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0

"""
#######################
Validate Model
#######################
"""
def validation(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num

"""
#######################
Main
#######################
"""

if __name__ == '__main__':
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create timer
    timer = Timer()
    # get args
    args = get_args()
    # Configure logger
    logging.basicConfig()
    # Set logger to print
    logging.root.setLevel(logging.NOTSET)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # Logging
    logging.info(args)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create transformations to use on image data set
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    # Load config
    config = config
    # Set datasets array
    datasets = []
    # Set number of classes to default
    num_classes = 1001
    # Set label file
    label_file = ''
    # Depending on configuration either load up VOC dataset or egohands dataset
    for dataset_path in args.datasets:
        if args.dataset_type == 'voc':
            dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform, keep_difficult=args.keep_difficult)
            label_file = os.path.join("/Users/Byakugan/GitHub/MobileNetV1-SSD/data/voc/VOC2012/", "voc-model-labels.txt")
            #store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        elif args.dataset_type == 'egohands':
            dataset = EgoHandsDataset(dataset_path,
                                        transform=train_transform, target_transform=target_transform,
                                      keep_difficult=args.keep_difficult)
            label_file = os.path.join("/Users/Byakugan/GitHub/MobileNetV1-SSD/data/egohands/egohandsVOC/", "ego-model-labels.txt")
            #store_labels(label_file, dataset.class_names)
            logging.info(dataset)
            num_classes = len(dataset.class_names)

        else:
            raise ValueError(f"Dataset type {args.dataset_type} is not supported.")
        datasets.append(dataset)
    # Load training dataset
    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    # Load validation dataset
    logging.info("Prepare Validation datasets.")
    if args.dataset_type == "voc":
        val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_validation=True, keep_difficult=args.keep_difficult)
    elif args.dataset_type == 'egohands':
        val_dataset = EgoHandsDataset(args.validation_dataset, transform=test_transform, keep_difficult=args.keep_difficult,
                                 target_transform=target_transform, is_validation=True)
        logging.info(val_dataset)
    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    logging.info("Build network.")
    # Create our mobilenetv1 model
    net = MobileNetV1_SSD(num_classes=num_classes, is_test=False)
    # Set loss
    min_loss = -10000.0
    # Initialise last epoch
    last_epoch = -1
    # Set learning rate for base network
    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    # Set learning rate for extra layers if different
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    # Decide whether to freeze the base network weights based upon config
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        #params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
        #                         net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    # Decide whether to freeze the entire network bar prediction heads
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else: # Set parameters for learning
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    # Load the mobilenetv1 model, depending on configuration either resume from past model, pretrained model or base model
    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')
    # Set to device
    net.to(device)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=device)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")
    # Set scheduler
    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        sys.exit(1)
    ############################
    # Start the training process
    ############################
    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):
        scheduler.step()
        train(train_loader, net, criterion, optimizer,
              device=device, debug_steps=args.debug_steps, epoch=epoch)

        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = validation(val_loader, net, criterion, device)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join(args.checkpoint_folder, f"mobilenetv1-SSD-Epoch-{epoch}-Loss-{val_loss}.pth")
            net.save(model_path)
            logging.info(f"Saved model {model_path}")











