"""Python"""
import os
import time
import sys
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import test  # import test.py to get mAP after each epoch
from models.models import *
from utils.datasets import *
from utils.utils import *
from utils.parse_config import *
from utils.misc import str2bool, Timer, freeze_net_layers


"""
#######################
Description
#######################

--config yolov3-spp-ego.cfg --data_config data/egohands/egohandsDarknet/egohands.data

This code is adapted from 'https://github.com/ultralytics/yolov3/'
"""


# Set mixed precision
mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

# Set weights directory
wdir = 'models/weights' + os.sep
# Set directory path for last set of weights
last = wdir + 'last.pt'
# Set directory path for best set of weights
best = wdir + 'best.pt'
# Set directory path for results
results_file = 'results.txt'
# Hyperparameters for run
hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # class loss gain
       'cls_pw': 1.0,  # class BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

# Print focal loss if gamma > 0
if hyp['fl_gamma']:
    print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])


"""
#######################
Arguments
#######################
"""
def get_args():
    # Set parsed arguments
    parser = argparse.ArgumentParser(
        description='YOLOv3 object detector')
    # Params for datasets
    parser.add_argument('--data_config', type=str, default='data/egohands/egohandsDarknet/egohands.data', help='Dataset configuration directory path')
    parser.add_argument('--multi_scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img_size', nargs='+', type=int, default=[320, 640], help='[min_train, max-train, test]')
    parser.add_argument('--cache_images', action='store_true', help='cache images for faster training')
    parser.add_argument('--single_class', action='store_true', help='train as single-class dataset')
    parser.add_argument('--freeze_net', action='store_true',
                        help="Freeze all the layers except the prediction head.")
    parser.add_argument('--freeze_layers', action='store_true', help='Freeze non-output layers')
    # Params for ADAM
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')

    """
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
    """
    # Params for loading pretrained basenet or checkpoints.
    parser.add_argument('--resume', action='store_true',
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--weights', type=str, default='models/weights/yolov3-spp-ultralytics.pt', help='initial weights path')
    # Params for model and run configuration
    parser.add_argument('--config', type=str,
                        default = 'config/yolov3-spp-ego.cfg', help = '*.cfg path')
    parser.add_argument('--no_save', action='store_true', help='only save final checkpoint')
    parser.add_argument('--no_test', action='store_true', help='only test final epoch')
    # Train params
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size for training') # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--num_epochs', default=200, type=int,
                        help='the number epochs')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    return parser.parse_args()


"""
#######################
Train Model
#######################
"""

def train(hyp):
    # Get configuration
    config = args.config
    # Get data configuration
    data = args.data_config
    # Get epochs
    epochs = args.num_epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    # Get bathc size
    batch_size = args.batch_size
    # Set accumlate
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    # Get initial training weights
    weights = args.weights  # initial training weights
    # Set image sizes
    imgsz_min, imgsz_max, imgsz_test = args.img_size  # img sizes (min, max, test)
    # Image grid sizes
    gs = 32  # (pixels) grid size
    # Check value
    assert math.fmod(imgsz_min, gs) == 0, '--img_size %g must be a %g-multiple' % (imgsz_min, gs)
    # Set scale
    args.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
    if args.multi_scale:
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.5
            imgsz_max //= 0.667
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
    # Set size
    img_size = imgsz_max  # initialize with max size
    # Configure run
    init_seeds()
    # Parse data configuration
    data_dict = parse_data_cfg(data)
    # Set paths
    train_path = data_dict['train']
    test_path = data_dict['valid']
    # Set number of classes
    number_classes = 1 if args.single_class else int(data_dict['classes'])  # number of classes
    hyp['cls'] *= number_classes / 80  # update coco-tuned hyp['cls'] to current dataset
    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)

    # Initialize model
    model = Darknet(config).to(device)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if args.adam:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2
    # Set epoch and fitness values
    start_epoch = 0
    best_fitness = 0.0
    # Atempt to download weights
    attempt_download(weights)
    # Check format and try and load model / checkpoint
    if weights.endswith('.pt'):  # pytorch format
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        ckpt = torch.load(weights, map_location=device)
        # load model
        try:
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
        except KeyError as e: # Check compatibility of weights with configuration expected
            s = "%s is not compatible with %s. Specify --weights '' or specify a --config compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (args.weights, args.config, args.weights)
            raise KeyError(s) from e
        # load optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
        # load results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt
        # epochs
        start_epoch = ckpt['epoch'] + 1
        if epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (args.weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs
        # Delete checkpoint model
        del ckpt
    # Load darknet format model weights
    elif len(weights) > 0:  # darknet format
        # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        load_darknet_weights(model, weights)
    # Check whether to freeze layer weights
    if args.freeze_layers:
        output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list) if
                                isinstance(module, YOLOLayer)]
        freeze_layer_indices = [x for x in range(len(model.module_list)) if
                                (x not in output_layer_indices) and
                                (x - 1 not in output_layer_indices)]
        for idx in freeze_layer_indices:
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1  # see link below
    # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822

    # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, '.-', label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=args.rect,  # rectangular training
                                  cache_images=args.cache_images,
                                  single_cls=args.single_class)

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if (args.num_workers != -1):
        # Do not automatically set workers, use supplied values
        nw = args.num_workers
    # Data loader
    dataloader = DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not args.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Testloader
    testloader = DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                                                 hyp=hyp,
                                                                 rect=True,
                                                                 cache_images=args.cache_images,
                                                                 single_cls=args.single_class),
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Model parameters
    model.nc = number_classes  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, number_classes).to(device)  # attach class weights

    # Model EMA
    ema = torch_utils.ModelEMA(model)

    # Start training
    nb = len(dataloader)  # number of batches
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
    maps = np.zeros(number_classes)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, imgsz_test))
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)
    # Begin training epochs
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=number_classes, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(4).to(device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            # Burn-in
            if ni <= n_burn:
                xi = [0, n_burn]  # x interp
                model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, 64 / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    x['weight_decay'] = np.interp(ni, xi, [0.0, hyp['weight_decay'] if j == 1 else 0.0])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])
            # Multi-Scale
            if args.multi_scale:
                if ni / accumulate % 1 == 0:  # adjust img_size (67% - 150%) every 1 batch
                    img_size = random.randrange(grid_min, grid_max + 1) * gs
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            pred = model(imgs)

            # Loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Backward
            loss *= batch_size / 64  # scale loss
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

            # Plot
            if ni < 1:
                f = 'train_batch%g.jpg' % i  # filename
                res = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                if tb_writer:
                    tb_writer.add_image(f, res, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()

        # Process epoch results
        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs
        if not args.no_test or final_epoch:  # Calculate mAP
            is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
            results, maps = test.test(config,
                                      data,
                                      batch_size=batch_size,
                                      imgsz=imgsz_test,
                                      model=ema.ema,
                                      save_json=final_epoch and is_coco,
                                      single_cls=args.single_class,
                                      dataloader=testloader,
                                      multi_label=ni > n_burn)

        # Write
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(args.name) and args.bucket:
            os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (args.bucket, args.name))

        # Tensorboard
        if tb_writer:
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        save = (not args.no_save) or (final_epoch and not args.evolve)
        if save:
            with open(results_file, 'r') as f:  # create checkpoint
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': f.read(),
                        'model': ema.ema.module.state_dict() if hasattr(model, 'module') else ema.ema.state_dict(),
                        'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last, best and delete
            torch.save(ckpt, last)
            if (best_fitness == fi) and not final_epoch:
                torch.save(ckpt, best)
            del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    # get name
    name = args.name
    if len(name):
        name = '_' + name if not name.isnumeric() else name
        fresults, flast, fbest = 'results%s.txt' % name, wdir + 'last%s.pt' % name, wdir + 'best%s.pt' % name
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                ispt = f2.endswith('.pt')  # is *.pt
                strip_optimizer(f2) if ispt else None  # strip optimizer
                os.system('gsutil cp %s gs://%s/weights' % (f2, args.bucket)) if args.bucket and ispt else None  # upload

    if not args.evolve:
        plot_results()  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

"""
#######################
Main
#######################
"""

if __name__ == '__main__':
    # Get args
    args = get_args()
    # Set weights to either load from base or last saved set if we are resuming training
    args.weights = last if args.resume and not args.weights else args.weights
    # Check arguments and data file paths
    args.config = check_file(args.config)  # check file
    args.data_config = check_file(args.data_config)  # check file
    # Print arguments for training
    print(args)
    # Extend
    args.img_size.extend([args.img_size[-1]] * (3 - len(args.img_size)))  # extend to 3 sizes (min, max, test)
    # Set device, either cuda or cpu
    device = torch_utils.select_device(args.device, apex=mixed_precision, batch_size=args.batch_size)
    # Set mixed precision to false if running on cpu
    if device.type == 'cpu':
        mixed_precision = False
    # Set writer
    tb_writer = None
    # Determine whether to train normally or evolve hyperparameters
    if not args.evolve:  # Train normally
        # Start tensorboard and log run
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(comment=args.name)
        # Train using supplied hyperparameters
        train(hyp)  # train normally

    else:  # Evolve hyperparameters (optional)
        args.no_test, args.no_save = True, True  # only test/save final epoch
        if args.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % args.bucket)  # download evolve.txt if exists

        for _ in range(1):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination
                # Mutate
                method, mp, s = 3, 0.9, 0.2  # method, mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                ng = len(g)
                if method == 1:
                    v = (npr.randn(ng) * npr.random() * g * s + 1) ** 2.0
                elif method == 2:
                    v = (npr.randn(ng) * npr.random(ng) * g * s + 1) ** 2.0
                elif method == 3:
                    v = np.ones(ng)
                    while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                        # v = (g * (npr.random(ng) < mp) * npr.randn(ng) * s + 1) ** 2.0
                        v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = x[i + 7] * v[i]  # mutate

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            results = train(hyp.copy())

            # Write mutation results
            print_mutation(hyp, results, args.bucket)

            # Plot results
            plot_evolution_results(hyp)








