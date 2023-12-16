import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import orca_models
import open_world_imagenet as datasets
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from utils import cluster_acc, AverageMeter, entropy, MarginLoss, accuracy, TransformTwice, degrade_data, Regularizer, MMD, save_checkpoint
from sklearn import metrics
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle
import tqdm
import json
from datetime import datetime
import glob

def train(args, model, device, train_loader, optimizer, m, labeled_len, epoch, tf_writer):
    model.train()
    bce = nn.BCELoss()
    m = min(m, 0.5)
    ce = MarginLoss(m=-1*m)
    bce_losses = AverageMeter('bce_loss', ':.4e')
    ce_losses = AverageMeter('ce_loss', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')
    cs_losses = AverageMeter('cs_loss', ':.4e')
    if args.mmd:
        mmd_losses = AverageMeter('mmd', ':.4e')

    for batch_idx, ((x, x2), combined_target, idx) in enumerate(train_loader):
        
        target = combined_target[:labeled_len]
        x, x2, target = x.to(device), x2.to(device), target.to(device)
        
        optimizer.zero_grad()

        if args.dsbn:
            output_s, feat_s = model(x[:labeled_len])
            output_t, feat_t = model(x[labeled_len:])
            output, feat = torch.cat((output_s, output_t)), torch.cat((feat_s, feat_t))
            
            output2_s, feat2_s = model(x2[:labeled_len])
            output2_t, feat2_t = model(x2[labeled_len:])
            output2, feat2 = torch.cat((output2_s, output2_t)), torch.cat((feat2_s, feat2_t))
        else:
            output, feat = model(x)
            output2, feat2 = model(x2)

        prob = F.softmax(output, dim=1)
        prob2 = F.softmax(output2, dim=1)

        # Similarity labels
        feat_detach = feat.detach()
        feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
        cosine_dist = torch.mm(feat_norm, feat_norm.t())
        labeled_len = len(target)

        pos_pairs = []
        target_np = target.cpu().numpy()
        # label part
        for i in range(labeled_len):
            target_i = target_np[i]
            idxs = np.where(target_np == target_i)[0]
            if len(idxs) == 1:
                pos_pairs.append(idxs[0])
            else:
                selec_idx = np.random.choice(idxs, 1)
                while selec_idx == i:
                    selec_idx = np.random.choice(idxs, 1)
                pos_pairs.append(int(selec_idx))
        # unlabel part
        unlabel_cosine_dist = cosine_dist[labeled_len:, :]
        vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
        pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
        pos_pairs.extend(pos_idx)

        # Clustering and consistency losses
        pos_prob = prob2[pos_pairs, :]
        pos_sim = torch.bmm(prob.view(args.batch_size, 1, -1), pos_prob.view(args.batch_size, -1, 1)).squeeze()
        ones = torch.ones_like(pos_sim)
        bce_loss = bce(pos_sim, ones)
        ce_loss = ce(output[:labeled_len], target)
        entropy_loss = entropy(torch.mean(prob, 0))

        # uncertainty term of unlabeled data
        if args.regularizer:

            var = torch.ones(len(prob)).to(device) - torch.max(prob, dim=1)[0] # consider margin loss as a proxy for variance
            var_fn = Regularizer.monotonic_fn(var)
            var_fn = torch.clamp((var_fn-torch.min(var_fn))/(torch.max(var_fn)-torch.min(var_fn)), min=1e-8, max=(1-1e-4))
            
            if args.weighted:
                ### Soft novel class rejection Domain Adaptation
                prob_known = prob[:, :args.labeled_num]
                prob_known = 1/prob_known.sum(1).unsqueeze(1).repeat(1, args.labeled_num) * prob_known
                ent_known = - (prob_known * torch.log(prob_known)).sum(dim=1)
                w = torch.exp(-ent_known) / torch.exp(-ent_known).sum()
                var_reg = (w[labeled_len:]*var_fn[labeled_len:]).sum()
            else:
                var_reg = torch.mean(var_fn[labeled_len:])

        if args.mmd:    
            # MMD in latent space Z
            feat_u = feat[labeled_len:]
            z_t = feat_u[torch.randperm(len(feat_u))[:labeled_len]]
            z_s = feat[:labeled_len]
            mmd_loss = MMD(z_s=z_s, z_t=z_t, kernel='rbf', device=device)

        # overall loss 
        loss = - entropy_loss + ce_loss + bce_loss
        if args.regularizer:
            phi = args.phi
            loss = loss + phi*(var_reg)
            if args.mmd:
                loss = loss + 0.5*phi*mmd_loss

        bce_losses.update(bce_loss.item(), args.batch_size)
        ce_losses.update(ce_loss.item(), args.batch_size)
        entropy_losses.update(entropy_loss.item(), args.batch_size)
        if args.regularizer:
            cs_losses.update(var_reg.item(), args.batch_size)
            if args.mmd:
                mmd_losses.update(mmd_loss.item(), args.batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tf_writer.add_scalar('loss/bce', bce_losses.avg, epoch)
    tf_writer.add_scalar('loss/ce', ce_losses.avg, epoch)
    tf_writer.add_scalar('loss/entropy', entropy_losses.avg, epoch)
    if args.regularizer:
        tf_writer.add_scalar('loss/regularizer', cs_losses.avg, epoch)
        tf_writer.add_scalar('variable/w_labeled', var_fn[:labeled_len].mean(), epoch)
        tf_writer.add_scalar('variable/w_unlabeled', var_fn[labeled_len:].mean(), epoch)
        if args.mmd:
            tf_writer.add_scalar('loss/mmd', mmd_losses.avg, epoch)


def test(args, model, labeled_num, device, test_loader, epoch, tf_writer):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label, _) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            output, _ = model(x)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
    targets = targets.astype(int)
    preds = preds.astype(int)
    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(preds, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    mean_uncert = 1 - np.mean(confs)
    print('Test overall acc {:.4f}, label acc {:.4f}, unlabel acc {:.4f}'.format(overall_acc, seen_acc, unseen_acc))
    tf_writer.add_scalar('acc/overall', overall_acc, epoch)
    tf_writer.add_scalar('acc/seen', seen_acc, epoch)
    tf_writer.add_scalar('acc/unseen', unseen_acc, epoch)
    tf_writer.add_scalar('nmi/unseen', unseen_nmi, epoch)
    tf_writer.add_scalar('uncert/test', mean_uncert, epoch)
    return mean_uncert

parser = argparse.ArgumentParser(
            description='orca',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='visda', choices=['imagenet100', 'visda'], help='dataset setting')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3)')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--milestones', nargs='+', type=int, default=[30, 60])
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--dataset_root', default='/home/hirokiwaida/data/imagenet/train/', type=str)
parser.add_argument('--exp_root', type=str, default='./results/')
parser.add_argument('--labeled-num', default=6, type=int)
parser.add_argument('--labeled-ratio', default=0.5, type=float)
parser.add_argument('--model_name', type=str, default='resnet')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--num_classes', default=12, type=int)
parser.add_argument('--regularizer', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--mmd', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--dsbn', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--phi', type=float, default=0.5)
parser.add_argument('--pretrained', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--weighted', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--covariate-shift', default=False, action='store_true', 
                help='boolean flag for whether covariate shift is present')
parser.add_argument('--degrade-level', type=int, default=0, choices=[0,1,2,3])
parser.add_argument('--degrade-choice', type=str, default='none', choices=['none', 'blur', 'jitter'],
                help='choice of dataset shift')

if __name__ == "__main__":

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.dataset == 'visda':
        args.dataset_root = '/home/voan/orca/datasets/visda/'
        args.num_classes = 12
    name = f'{args.dataset}'
    if args.dataset != 'visda':
        name = os.path.join(name, f'{args.degrade_choice}{str(args.degrade_level)}')
    if args.regularizer:
        if not args.mmd:
            name = name + '/reg2'
        else:
            name = name + '/reg3'
    else:
        name = name + '/no_reg'
    args.savedir = os.path.join(args.exp_root, name, datetime.now().strftime("%m.%d.%H.%M.%S") + '_' + str(args.phi))
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    args.savedir = args.savedir + '/'

    if args.covariate_shift:
        degrade = degrade_data(degrade_level=args.degrade_level, degrade_choice=args.degrade_choice)
    
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    # if args.dataset == 'imagenet100':
    #     model = orca_models.resnet50(num_classes=args.num_classes)
    # elif args.dataset == 'visda':
    #     model = orca_models.resnet18(num_classes=args.num_classes)
    model = orca_models.resnet50(num_classes=args.num_classes)
    
    if args.pretrained:
        # pretrained_path = os.path.join('/home/voan/SimCLR/runs', args.dataset, args.degrade_choice + str(args.degrade_level))
        # if not args.regularizer:
        #     pretrained_path = os.path.join(pretrained_path, 'noreg')
        # pretrained_path = os.path.join(pretrained_path, '*')
        # pretrained_path = sorted(glob.glob(pretrained_path))[1]
        # pretrained_path = os.path.join(pretrained_path, 'checkpoint_{:04d}.pth.tar'.format(args.epochs))

        # checkpoint = torch.load(pretrained_path)
        # state_dict = checkpoint['state_dict'] 

        # for k in list(state_dict.keys()):
        #     if k.startswith('backbone.'):
        #         # remove prefix
        #         if k.startswith('backbone') and not k.startswith('backbone.fc'):
        #             state_dict[k[len("backbone."):]] = state_dict[k]
        #         if k.startswith('backbone.fc'):
        #             state_dict['contrastive_head.' + k[len("backbone.fc."):]] = state_dict[k]
        #     del state_dict[k]
        # # elif args.degrade_choice == 'none':
        # #     cov_shift_dl1 = '/home/voan/SimCLR/runs/May12_10-30-00_deepbox/checkpoint_0200.pth.tar'
        # model.load_state_dict(state_dict, strict=False)

        state_dict = torch.load('/home/voan/orca/pretrained/simclr_imagenet_100.pth.tar')
        model.load_state_dict(state_dict, strict=False)

    for name, param in model.named_parameters(): 
        if 'fc' not in name and 'layer4' not in name:
            param.requires_grad = False
    
    model = model.to(device)
    transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    transform_visda_source = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                # transforms.Normalize(mean=[0.8634, 0.8608, 0.8570], std=[0.2281, 0.2336, 0.2418])
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    transform_visda_target = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
                # transforms.Normalize(mean=[0.3682, 0.3451, 3265], std=[0.2909, 0.2798, 0.2786])
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    if args.dataset == 'imagenet100':
        train_label_set = datasets.ImageNetDataset(root=args.dataset_root, anno_file='/home/voan/orca/data/ImageNet100_label_{}_{:.1f}.txt'.format(args.labeled_num, args.labeled_ratio), transform=TransformTwice(transform_train))
        if args.covariate_shift:
            train_unlabel_set = datasets.ImageNetDataset(root=args.dataset_root, anno_file='/home/voan/orca/data/ImageNet100_unlabel_{}_{:.1f}.txt'.format(args.labeled_num, args.labeled_ratio), transform=TransformTwice(transforms.Compose([transform_train, degrade])))
            test_unlabel_set = datasets.ImageNetDataset(root=args.dataset_root, anno_file='/home/voan/orca/data/ImageNet100_unlabel_50_0.5.txt', transform=transforms.Compose([transform_test, degrade]))
        else:
            train_unlabel_set = datasets.ImageNetDataset(root=args.dataset_root, anno_file='/home/voan/orca/data/ImageNet100_unlabel_{}_{:.1f}.txt'.format(args.labeled_num, args.labeled_ratio), transform=TransformTwice(transform_train))
            test_unlabel_set = datasets.ImageNetDataset(root=args.dataset_root, anno_file='/home/voan/orca/data/ImageNet100_unlabel_50_0.5.txt', transform=transform_test)


    elif args.dataset == 'visda':
        train_label_set = datasets.VisDA(root=args.dataset_root, anno_file='/home/voan/orca/data/visda_label_{}_{:.1f}.txt'.format(args.labeled_num, args.labeled_ratio), transform=TransformTwice(transform_visda_source))
        train_unlabel_set = datasets.VisDA(root=args.dataset_root, anno_file='/home/voan/orca/data/visda_unlabel_{}_{:.1f}.txt'.format(args.labeled_num, args.labeled_ratio), source='t', transform=TransformTwice(transform_visda_source))
        test_unlabel_set = datasets.VisDA(root=args.dataset_root, anno_file='/home/voan/orca/data/visda_unlabel_{}_{:.1f}.txt'.format(args.labeled_num, args.labeled_ratio), source='t', transform=transform_visda_target)

    concat_set = datasets.ConcatDataset((train_label_set, train_unlabel_set))
    labeled_idxs = range(len(train_label_set)) 
    unlabeled_idxs = range(len(train_label_set), len(train_label_set)+len(train_unlabel_set))
    batch_sampler = datasets.TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, int(args.batch_size * len(train_unlabel_set) / (len(train_label_set) + len(train_unlabel_set))))

    train_loader = torch.utils.data.DataLoader(concat_set, batch_sampler=batch_sampler, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_unlabel_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    # Set the optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    tf_writer = SummaryWriter(log_dir=args.savedir)

    for epoch in tqdm.tqdm(range(args.epochs), total=args.epochs):
        mean_uncert = test(args, model, args.labeled_num, device, test_loader, epoch, tf_writer)
        train(args, model, device, train_loader, optimizer, mean_uncert, batch_sampler.primary_batch_size, epoch, tf_writer)
        scheduler.step()
    
    # save model checkpoints
    checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(args.epochs)
    save_checkpoint({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=False, filename=os.path.join(tf_writer.log_dir, checkpoint_name))
