'''
Finetune auxiliary classification head.
'''
import argparse, os, warnings, datetime
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
import torchvision.transforms as transforms

from datasets.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100
from datasets.ImbalanceImageNet import LT_Dataset
from models.our_resnet import ResNet18, ResNet34
from models.our_resnet_imagenet import ResNet50

from utils.utils import *
from utils.ltr_metrics import *

def get_args_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='Auxiliar branch finetuning.')
    parser.add_argument('--gpu', default='3')
    parser.add_argument('--num_workers', '--cpus', default=8, help='number of threads for data loader')
    parser.add_argument('--data_root_path', '--drp', default='/ssd1/haotao/datasets', help='data root path')
    parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--id_class_number', type=int, default=1000, help='for ImageNet subset')
    parser.add_argument('--model', '--md', default='ResNet18', choices=['ResNet18', 'ResNet34', 'ResNet50'], help='which model to use')
    parser.add_argument('--imbalance_ratio', '--rho', default=0.01, type=float)
    # training params:
    parser.add_argument('--odc',  type=int, default=30, help='')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='input batch size for training')
    parser.add_argument('--test_batch_size', '--tb', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--opt', default='adam', choices=['sgd', 'adam'], help='which optimizer to use')
    parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='which lr decay method to use')
    parser.add_argument('--decay_epochs', '--de', default=[1,2], nargs='+', type=int, help='milestones for multisteps lr decay')
    parser.add_argument('--num_ood_samples', default=30000, type=float, help='Number of OOD samples to use.')
    parser.add_argument('--tau0', type=float, default=1, help='logit adjustment hyper-parameter')
    parser.add_argument('--tau1', type=float, default=1, help='logit adjustment hyper-parameter')
    parser.add_argument('--tau2', type=float, default=1, help='logit adjustment hyper-parameter')
    # 
    parser.add_argument('--layer', default='both', choices=['both', 'fc', 'bn'], 
        help='Which layers to finetune. both: both BN and the last fc layers | bn: all bn layers excluding fc layer | fc: the final fc layer excluding all bn layers')
    parser.add_argument('--pretrained_exp_str', default='e200-b256-adam-lr0.001-wd0.0005-cos_Lambda0.5', help='Starting point for finetuning')
    parser.add_argument('--save_root_path', '--srp', default='/ssd1/haotao/', help='data root path')
    #
    parser.add_argument('--timestamp', action='store_true', help='If true, attack time stamp after exp str')
    # ddp 
    parser.add_argument('--ddp', action='store_true', help='If true, use distributed data parallel')
    parser.add_argument('--ddp_backend', '--ddpbed', default='nccl', choices=['nccl', 'gloo', 'mpi'], help='If true, use distributed data parallel')
    parser.add_argument('--num_nodes', default=1, type=int, help='Number of nodes')
    parser.add_argument('--node_id', default=0, type=int, help='Node ID')
    parser.add_argument('--dist_url', default='tcp://localhost:23456', type=str, help='url used to set up distributed training')
    args = parser.parse_args()

    return args

def create_save_path():
    # mkdirs:
    opt_str = 'e%d-b%d-%s-lr%s-wd%s-cos' % (args.epochs, args.batch_size, args.opt, args.lr, args.wd)
    loss_str = 'LA-tau%s%s%s-Lambda%s' % (args.tau0,args.tau1,args.tau2,args.Lambda)
    exp_str = '%s_%s' % (opt_str, loss_str) if loss_str else opt_str
    exp_str = 'finetune_%s_' % args.layer + exp_str
    if args.timestamp:
        exp_str += '_%s' % datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    dataset_str = '%s-%s-OOD%d' % (args.dataset, args.imbalance_ratio, args.num_ood_samples) if 'imagenet' not in args.dataset else '%s%d-lt' % (args.dataset, args.id_class_number)
    save_dir = os.path.join(args.save_root_path, 'LT_OOD_results', 'EAT', dataset_str, args.model, args.pretrained_exp_str, exp_str)
    print('Saving to %s' % save_dir)
    create_dir(save_dir)

    return save_dir

def setup(rank, ngpus_per_node, args):
    # initialize the process group
    world_size = ngpus_per_node * args.num_nodes
    dist.init_process_group(args.ddp_backend, init_method=args.dist_url, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def set_mode_to_train(model, args, fp=None):
    if fp is not None:
        fp.write('All modules in training mode:\n')
    for name, m in model.named_modules():
        if len(list(m.children())) > 0: # not a leaf node
            continue
        if args.layer == 'all':
            m.train()
        else:
            m.eval()
        if args.layer == 'both':
            if any([_name in name for _name in ['bn', 'shortcut.1', 'downsample.1']]):
                m.train()
        elif args.layer == 'bn':
            if any([_name in name for _name in ['bn', 'shortcut.1', 'downsample.1']]):
                m.train()
        # print(name, m.training)
        if m.training and fp is not None:
            fp.write(name+'\n')

def train(gpu_id, ngpus_per_node, args):

    save_dir = args.save_dir

    fp = open(os.path.join(save_dir, 'train_log.txt'), 'a+')
    fp_val = open(os.path.join(save_dir, 'val_log.txt'), 'a+')

    # get globale rank (thread id):
    rank = args.node_id * ngpus_per_node + gpu_id

    print(f"Running on rank {rank}.")

    # Initializes ddp:
    if args.ddp:
        setup(rank, ngpus_per_node, args)

    # intialize device:
    device = gpu_id if args.ddp else 'cuda'
    torch.backends.cudnn.benchmark = True

    # get batch size:
    train_batch_size = args.batch_size if not args.ddp else int(args.batch_size/ngpus_per_node/args.num_nodes)
    num_workers = args.num_workers if not args.ddp else int((args.num_workers+ngpus_per_node)/ngpus_per_node)

    # data:
    if args.dataset in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    odc = args.odc
    if args.dataset == 'cifar10':
        num_classes = 10 + odc
        train_set = IMBALANCECIFAR10(train=True, transform=train_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR10(train=False, transform=test_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
    elif args.dataset == 'cifar100':
        num_classes = 100 + odc
        train_set = IMBALANCECIFAR100(train=True, transform=train_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR100(train=False, transform=test_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
    elif args.dataset == 'imagenet':
        num_classes = args.id_class_number
        train_set = LT_Dataset(
            os.path.join(args.data_root_path, 'imagenet'), './datasets/ImageNet_LT/ImageNet_LT_train.txt', transform=train_transform, 
            subset_class_idx=np.arange(0,args.id_class_number))
        test_set = LT_Dataset(
            os.path.join(args.data_root_path, 'imagenet'), './datasets/ImageNet_LT/ImageNet_LT_val.txt', transform=test_transform,
            subset_class_idx=np.arange(0,args.id_class_number))
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=not args.ddp, num_workers=num_workers,
                                drop_last=True, pin_memory=True, sampler=train_sampler)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers, 
                                drop_last=False, pin_memory=True)
    print('Training on %s with %d images and %d validation images.' % (args.dataset, len(train_set), len(test_set)))
    # get prior distributions:
    img_num_per_cls = np.array(train_set.img_num_per_cls)
    extendlist = [ 1000 for i in range(odc) ]
    img_num_per_cls1 = np.append(extendlist, img_num_per_cls)
    prior = img_num_per_cls1 / np.sum(img_num_per_cls1)
    prior = torch.from_numpy(prior).float().to(device)

    # model:
    if args.model == 'ResNet18':
        model = ResNet18(num_classes=num_classes).to(device)
    elif args.model == 'ResNet34':
        model = ResNet34(num_classes=num_classes).to(device)
    elif args.model == 'ResNet50':
        model = ResNet50(num_classes=num_classes).to(device)
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], broadcast_buffers=False, find_unused_parameters=True)
    else:
        # model = torch.nn.DataParallel(model)
        pass

    pretrained_ckpt_path = os.path.join(save_dir, '../', 'latest.pth')
    print('Loading ckpt from %s' % pretrained_ckpt_path)
    ckpt = torch.load(pretrained_ckpt_path)
    model.load_state_dict(ckpt['model'], strict=False)
    # only update bn and fc layers:
    # WARNING: All parameters with requires_grad=True should participate in loss calculation or DDP will complain. 
    fp.write('ALl parameters requiring grad:'+'\n')    
    for name, p in model.named_parameters(): 
        # if args.layer == 'all':
        #     p.requires_grad = True 
        # else:
        #     p.requires_grad = False 
        # if args.layer == 'both':
        #     if any([_name in name for _name in ['bn', 'shortcut.1', 'downsample.1']]): 
        #         p.requires_grad = True
        #     if 'fc' in name or 'linear' in name:
        #         p.requires_grad = True 
        # elif args.layer == 'bn':
        #     if any([_name in name for _name in ['bn', 'shortcut.1', 'downsample.1']]):
        #         p.requires_grad = True
        # elif args.layer == 'fc':
        #     if 'fc' in name or 'linear' in name:
        #         p.requires_grad = True 
        p.requires_grad = False
        if any([_name in name for _name in ['linear_classfier']]):
            p.requires_grad = True
        #if any([_name in name for _name in ['bn']]):
        #    p.eval() 
        if p.requires_grad:
            print(name)
            fp.write(name+'\n')
        
    #set_mode_to_train(model, args, fp)


    # optimizer:
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # train:
    training_losses, training_accs, test_clean_losses = [], [], []
    f1s, overall_accs, many_accs, median_accs, low_accs = [], [], [], [], []
    overall_accs0, many_accs0, median_accs0, low_accs0 = [], [], [], []
    overall_accs1, many_accs1, median_accs1, low_accs1 = [], [], [], []
    overall_accs2, many_accs2, median_accs2, low_accs2 = [], [], [], []
    best_overall_acc = 0
    start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        # reset sampler when using ddp:
        if args.ddp:
            train_sampler.set_epoch(epoch)
        model.train()    
        set_mode_to_train(model, args)
        train_acc_meter, training_loss_meter = AverageMeter(), AverageMeter()
        for batch_idx, (in_data, labels) in enumerate(train_loader):
            in_data, labels = in_data.to(device), labels.to(device)
            in_labels = labels  +odc

            # forward:
            if args.dataset == 'cifar10' :
                head_idx0 = (labels>=odc) & (labels<odc+4)
                tail_idx0 = labels>=odc+6
                head_num = torch.nonzero((labels>=odc) & (labels<odc+4)).shape[0]
                tail_num = torch.nonzero(labels>=odc+6).shape[0]
            elif args.dataset == 'cifar100' :
                head_idx0 = (labels>=odc) & (labels<odc+40)
                tail_idx0 = labels>=odc+60
                head_num = torch.nonzero((labels>=odc) & (labels<odc+40)).shape[0]
                tail_num = torch.nonzero(labels>=odc+60).shape[0]
            elif args.dataset == 'imagenet' :
                head_idx0 = (labels>=odc) & (labels<odc+500)
                tail_idx0 = labels>=odc+500
                head_num = torch.nonzero((labels>=odc) & (labels<odc+500)).shape[0]
                tail_num = torch.nonzero(labels>=odc+500).shape[0]

            if tail_num != 0:
                times = head_num // tail_num
            
                tail = in_data[tail_idx0]
                tail_label = in_labels[tail_idx0]
                tail_data = tail
                tail_labels = tail_label
                for i in range(times-1):
                    tail_data = torch.cat([tail_data, tail], dim=0)
                    tail_labels = torch.cat([tail_labels, tail_label],dim=0)
                extra_data = in_data[head_idx0][:tail_labels.size(0)]

                lam = np.random.beta(0.9999, 0.9999)
                bbx1, bby1, bbx2, bby2 = rand_bbox(tail_data.size(), lam)
                extra_data[:, :, bbx1:bbx2, bby1:bby2] = tail_data[:, :, bbx1:bbx2, bby1:bby2]

                in_data = torch.cat([in_data, extra_data], dim=0)

                all_logits0, all_logits1, all_logits2 = model(in_data)

                adjusted_all_logits0 = all_logits0+ args.tau0 * prior.log()[None,:]
                lt_loss0 = F.cross_entropy(adjusted_all_logits0[:train_batch_size], in_labels) #+ \

                adjusted_all_logits1 = all_logits1 + args.tau1 * prior.log()[None,:]
                lt_loss1 = F.cross_entropy(adjusted_all_logits1[:train_batch_size], in_labels) #+ \
                        #args.Lambda * F.cross_entropy(adjusted_all_logits1[train_batch_size:], tail_labels)

                adjusted_all_logits2 = all_logits2 + args.tau2 * prior.log()[None,:]
                lt_loss2 = F.cross_entropy(adjusted_all_logits2[:train_batch_size], in_labels) #+ \
                        #args.Lambda * F.cross_entropy(adjusted_all_logits2[train_batch_size:], tail_labels)
            else :
                all_logits0, all_logits1, all_logits2, _ = model(in_data)
                # adjust logits:
                adjusted_all_logits0 = all_logits0 + args.tau0 * prior.log()[None,:]
                lt_loss0 = F.cross_entropy(adjusted_all_logits0, in_labels)
                adjusted_all_logits1 = all_logits1 + args.tau1 * prior.log()[None,:]
                lt_loss1 = F.cross_entropy(adjusted_all_logits1, in_labels) 
                adjusted_all_logits2 = all_logits2 + args.tau2 * prior.log()[None,:]
                lt_loss2 = F.cross_entropy(adjusted_all_logits2, in_labels) 

            #loss = in_loss
            loss = lt_loss0 + lt_loss1 + lt_loss2


            # adjusted_all_logits0 = all_logits0 + 1 * prior.log()[None,:]
            # lt_loss0 = F.cross_entropy(adjusted_all_logits0, in_labels)
            # med_tail_idx = (in_labels>=round( (num_classes-odc)/3) +odc)
            # adjusted_all_logits1 = all_logits1 + 1 * prior.log()[None,:]
            # lt_loss1 = F.cross_entropy(adjusted_all_logits1[med_tail_idx], in_labels[med_tail_idx])
            # tail_idx = (in_labels>=round(2*(num_classes-odc)/3)+odc)
            # adjusted_all_logits2 = all_logits2 + 1 * prior.log()[None,:]
            # lt_loss2 = F.cross_entropy(adjusted_all_logits2[tail_idx], in_labels[tail_idx])
            # #loss = in_loss
            # loss = lt_loss0 + 0.6*lt_loss1 + 0.1*lt_loss2

            # backward:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # append:
            training_loss_meter.append(loss.item())
            acc = ((F.softmax(all_logits0[:train_batch_size],dim=1)+F.softmax(all_logits1[:train_batch_size],dim=1)+F.softmax(all_logits2[:train_batch_size],dim=1))[:,odc:].argmax(1) == labels).float().mean()
            train_acc_meter.append(acc.item())

            if rank == 0 and batch_idx % 100 == 0:
                train_str = 'epoch %d batch %d (train): loss %.4f' % (epoch, batch_idx, loss.item()) 
                print(train_str)
                fp.write(train_str + '\n')
                fp.flush()

        # lr update:
        scheduler.step()

        if rank == 0:
            # eval on clean set:
            model.eval()
            test_acc_meter, test_loss_meter = AverageMeter(), AverageMeter()
            preds_list, labels_list = [], []
            preds_list0, preds_list1, preds_list2 = [], [], []
            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(device), labels.to(device)
                    logits0, logits1, logits2 = model(data)
                    pred = (F.softmax(logits0,dim=1)+F.softmax(logits1,dim=1)+F.softmax(logits2,dim=1))[:,odc:].argmax(dim=1, keepdim=True)
                    pred0 = logits0[:,odc:].argmax(dim=1, keepdim=True)
                    pred1 = logits1[:,odc:].argmax(dim=1, keepdim=True)
                    pred2 = logits2[:,odc:].argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    loss = F.cross_entropy(logits1, labels)
                    #test_acc_meter.append((logits.argmax(1) == labels).float().mean().item())
                    test_loss_meter.append(loss.item())
                    preds_list.append(pred)
                    preds_list0.append(pred0)
                    preds_list1.append(pred1)
                    preds_list2.append(pred2)
                    labels_list.append(labels)

            preds = torch.cat(preds_list, dim=0).detach().cpu().numpy().squeeze()
            preds0 = torch.cat(preds_list0, dim=0).detach().cpu().numpy().squeeze()
            preds1 = torch.cat(preds_list1, dim=0).detach().cpu().numpy().squeeze()
            preds2 = torch.cat(preds_list2, dim=0).detach().cpu().numpy().squeeze()
            labels = torch.cat(labels_list, dim=0).detach().cpu().numpy()

            overall_acc= (preds == labels).sum().item() / len(labels)
            overall_acc0= (preds0 == labels).sum().item() / len(labels)
            overall_acc1= (preds1 == labels).sum().item() / len(labels)
            overall_acc2= (preds2 == labels).sum().item() / len(labels)
            f1 = f1_score(labels, preds, average='macro')

            many_acc, median_acc, low_acc, _ = shot_acc(preds, labels, img_num_per_cls, acc_per_cls=True)
            many_acc0, median_acc0, low_acc0, _ = shot_acc(preds0, labels, img_num_per_cls, acc_per_cls=True)
            many_acc1, median_acc1, low_acc1, _ = shot_acc(preds1, labels, img_num_per_cls, acc_per_cls=True)
            many_acc2, median_acc2, low_acc2, _ = shot_acc(preds2, labels, img_num_per_cls, acc_per_cls=True)

            test_clean_losses.append(test_loss_meter.avg)
            f1s.append(f1)
            overall_accs.append(overall_acc)
            many_accs.append(many_acc)
            median_accs.append(median_acc)
            low_accs.append(low_acc)

            overall_accs0.append(overall_acc0)
            many_accs0.append(many_acc0)
            median_accs0.append(median_acc0)
            low_accs0.append(low_acc0)

            overall_accs1.append(overall_acc1)
            many_accs1.append(many_acc1)
            median_accs1.append(median_acc1)
            low_accs1.append(low_acc1)

            overall_accs2.append(overall_acc2)
            many_accs2.append(many_acc2)
            median_accs2.append(median_acc2)
            low_accs2.append(low_acc2)

            training_acc = train_acc_meter.avg
            val_str = 'epoch %d (test): ACC %.4f (%.4f, %.4f, %.4f) | F1 %.4f | (train) ACC %.4f' % (epoch, overall_acc, many_acc, median_acc, low_acc, f1, training_acc) 
            print(val_str)
            fp_val.write(val_str + '\n')
            fp_val.flush()

            # save curves:
            training_losses.append(training_loss_meter.avg)
            plt.plot(training_losses, 'b', label='training_losses')
            plt.plot(test_clean_losses, 'g', label='test_clean_losses')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'losses.png'))
            plt.close()

            training_accs.append(training_acc)
            plt.plot(training_accs, 'black', label='training_accs')
            plt.plot(overall_accs, 'm', label='overall_accs')
            if args.imbalance_ratio < 1:
                plt.plot(many_accs, 'r', label='many_accs')
                plt.plot(median_accs, 'g', label='median_accs')
                plt.plot(low_accs, 'b', label='low_accs')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'test_accs.png'))
            plt.close()

            plt.plot(overall_accs0, 'm', label='overall_accs')
            if args.imbalance_ratio < 1:
                plt.plot(many_accs0, 'r', label='many_accs')
                plt.plot(median_accs0, 'g', label='median_accs')
                plt.plot(low_accs0, 'b', label='low_accs')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'test_accs0.png'))
            plt.close()

            plt.plot(overall_accs1, 'm', label='overall_accs')
            if args.imbalance_ratio < 1:
                plt.plot(many_accs1, 'r', label='many_accs')
                plt.plot(median_accs1, 'g', label='median_accs')
                plt.plot(low_accs1, 'b', label='low_accs')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'test_accs1.png'))
            plt.close()

            plt.plot(overall_accs2, 'm', label='overall_accs')
            if args.imbalance_ratio < 1:
                plt.plot(many_accs2, 'r', label='many_accs')
                plt.plot(median_accs2, 'g', label='median_accs')
                plt.plot(low_accs2, 'b', label='low_accs')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'test_accs2.png'))
            plt.close()

            plt.plot(f1s, 'm', label='f1s')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'test_f1s.png'))
            plt.close()

            # save best model:
            if training_accs[-1] > best_overall_acc:
                best_overall_acc = training_accs[-1]
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_clean_acc.pth'))


            # save pth:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch, 
                'best_overall_acc': best_overall_acc,
                'training_losses': training_losses, 
                'test_clean_losses': test_clean_losses, 
                'f1s': f1s, 
                'overall_accs': overall_accs, 
                'many_accs': many_accs, 
                'median_accs': median_accs, 
                'low_accs': low_accs, 
                'overall_accs0': overall_accs0, 
                'many_accs0': many_accs0, 
                'median_accs0': median_accs0, 
                'low_accs0': low_accs0,
                'overall_accs1': overall_accs1, 
                'many_accs1': many_accs1, 
                'median_accs1': median_accs1, 
                'low_accs1': low_accs1,
                'overall_accs2': overall_accs2, 
                'many_accs2': many_accs2, 
                'median_accs2': median_accs2, 
                'low_accs2': low_accs2,
                }, 
                os.path.join(save_dir, 'latest.pth'))

    # Clean up ddp:
    if args.ddp:
        cleanup()

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

if __name__ == '__main__':
    # get args:
    args = get_args_parser()

    # mkdirs:
    save_dir = create_save_path()
    args.save_dir = save_dir
    
    # set CUDA:
    if args.num_nodes == 1: # When using multiple nodes, we assume all gpus on each node are available.
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 

    if args.ddp:
        ngpus_per_node = torch.cuda.device_count()
        torch.multiprocessing.spawn(train, args=(ngpus_per_node,args), nprocs=ngpus_per_node, join=True)
    else:
        train(0, 0, args)
