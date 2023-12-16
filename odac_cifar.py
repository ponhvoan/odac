import argparse
import json
import warnings
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import odac_models
import open_world_cifar as datasets
from utils.misc import cluster_acc, AverageMeter, entropy, MarginLoss, accuracy, TransformTwice, degrade_data, softmax_entropy, save_checkpoint
from utils.visualization_metrics import visualization
from sklearn import metrics
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle


def train(args, model, device, train_label_loader, train_unlabel_loader, optimizer, m, epoch, tf_writer):
    model.train()
    bce = nn.BCELoss()
    m = min(m, 0.5)
    ce = MarginLoss(m=-1*m)
    unlabel_loader_iter = cycle(train_unlabel_loader)
    bce_losses = AverageMeter('bce_loss', ':.4e')
    ce_losses = AverageMeter('ce_loss', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')
    cs_losses = AverageMeter('cs_loss', ':.4e')

    for batch_idx, ((x, x2, x_orig), target) in enumerate(train_label_loader):
        
        ((ux, ux2, ux_orig), _, index_t) = next(unlabel_loader_iter)

        x = torch.cat([x, ux], 0)
        x2 = torch.cat([x2, ux2], 0)
        labeled_len = len(target)

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
        
        pos_prob = prob2[pos_pairs, :]
        pos_sim = torch.bmm(prob.view(args.batch_size, 1, -1), pos_prob.view(args.batch_size, -1, 1)).squeeze()
        ones = torch.ones_like(pos_sim)
        bce_loss = bce(pos_sim, ones)
        ce_loss = ce(output[:labeled_len], target)
        entropy_loss = entropy(torch.mean(prob, 0))
        
        ### Soft novel class rejection Domain Adaptation
        prob_known = prob[:, :args.labeled_num]
        prob_known = 1/prob_known.sum(1).unsqueeze(1).repeat(1, args.labeled_num) * prob_known
        ent_known = - (prob_known * torch.log(prob_known)).sum(dim=1)
        w = torch.exp(-ent_known) / torch.exp(-ent_known).sum()
        
        # uncertainty term of unlabeled data
        w_ent = (w[labeled_len:]*softmax_entropy(output[labeled_len:])).mean(0) # w_ent

        loss = - entropy_loss + ce_loss + bce_loss + args.gamma*w_ent

        cs_losses.update(w_ent.item(), args.batch_size)
        bce_losses.update(bce_loss.item(), args.batch_size)
        ce_losses.update(ce_loss.item(), args.batch_size)
        entropy_losses.update(entropy_loss.item(), args.batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tf_writer.add_scalar('loss/bce', bce_losses.avg, epoch)
    tf_writer.add_scalar('loss/ce', ce_losses.avg, epoch)
    tf_writer.add_scalar('loss/entropy', entropy_losses.avg, epoch)
    tf_writer.add_scalar('loss/w_ent', cs_losses.avg, epoch)

    if args.vis:
        if epoch == 0:
            visualization(source_data=feat[:labeled_len].cpu().detach().numpy(), target_data=feat[labeled_len:].cpu().detach().numpy(), analysis='pca', data_name='cifar10_pretrain', save_path=args.save_vis)
            visualization(source_data=feat[:labeled_len].cpu().detach().numpy(), target_data=feat[labeled_len:].cpu().detach().numpy(), analysis='tsne', data_name='cifar10_pretrain', save_path=args.save_vis)
        if epoch == args.epochs-1:
            visualization(source_data=feat[:labeled_len].cpu().detach().numpy(), target_data=feat[labeled_len:].cpu().detach().numpy(), analysis='pca', data_name='cifar10_postrain', save_path=args.save_vis)
            visualization(source_data=feat[:labeled_len].cpu().detach().numpy(), target_data=feat[labeled_len:].cpu().detach().numpy(), analysis='tsne', data_name='cifar10_postrain', save_path=args.save_vis)

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
    print('Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(overall_acc, seen_acc, unseen_acc))
    tf_writer.add_scalar('acc/overall', overall_acc, epoch)
    tf_writer.add_scalar('acc/seen', seen_acc, epoch)
    tf_writer.add_scalar('acc/unseen', unseen_acc, epoch)
    tf_writer.add_scalar('nmi/unseen', unseen_nmi, epoch)
    tf_writer.add_scalar('uncert/test', mean_uncert, epoch)
    return mean_uncert, overall_acc, seen_acc, unseen_acc


def main():
    parser = argparse.ArgumentParser(description='orca')
    parser.add_argument('--milestones', nargs='+', type=int, default=[140, 180])
    parser.add_argument('--dataset', default='cifar100', help='dataset setting')
    parser.add_argument('--labeled-num', default=50, type=int)
    parser.add_argument('--labeled-ratio', default=0.5, type=float)
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--exp_root', type=str, default='./results/')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size')
    parser.add_argument('--pretrained', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--dsbn', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--covariate-shift', default=True, action=argparse.BooleanOptionalAction, 
                    help='boolean flag for whether covariate shift is present')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--degrade-level', type=int, default=1, choices=[0,1,2,3])
    parser.add_argument('--degrade-choice', type=str, default='blur', choices=['none', 'blur', 'jitter', 'elastic', 'glass_blur', 'snow', 'gaussian_noise'],
                    help='choice of dataset shift')
    parser.add_argument('--vis', action='store_true')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    args.savedir = os.path.join(args.exp_root, args.name)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    
    if (args.covariate_shift) & ('-c' not in args.dataset):
        degrade = degrade_data(degrade_level=args.degrade_level, degrade_choice=args.degrade_choice)

    if 'cifar10' in args.dataset:
        train_label_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train'], True))
        if args.dataset == 'cifar10':
            if args.covariate_shift:
                train_unlabel_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(transforms.Compose([datasets.dict_transform['cifar_train'], degrade]), True), unlabeled_idxs=train_label_set.unlabeled_idxs)
                test_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=transforms.Compose([datasets.dict_transform['cifar_test'], degrade]), unlabeled_idxs=train_label_set.unlabeled_idxs)
            else:
                train_unlabel_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train'], True), unlabeled_idxs=train_label_set.unlabeled_idxs)
                test_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=datasets.dict_transform['cifar_test'], unlabeled_idxs=train_label_set.unlabeled_idxs)
        elif args.dataset == 'cifar10-c':
            train_raw = np.load(f'./datasets/cifar/CIFAR-10-C/{args.degrade_choice}.npy')
            train_unlabel_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train'], True), unlabeled_idxs=train_label_set.unlabeled_idxs)
            train_unlabel_set.data = train_raw[train_label_set.unlabeled_idxs]
            test_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=datasets.dict_transform['cifar_test'], unlabeled_idxs=train_label_set.unlabeled_idxs)
            test_set.data = train_raw[train_label_set.unlabeled_idxs]
        num_classes = 10

    elif 'cifar100' in args.dataset:
        train_label_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train'], True))
        if args.dataset == 'cifar100':
            if args.covariate_shift:
                train_unlabel_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(transforms.Compose([datasets.dict_transform['cifar_train'], degrade]), True), unlabeled_idxs=train_label_set.unlabeled_idxs)
                test_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=transforms.Compose([datasets.dict_transform['cifar_test'], degrade]), unlabeled_idxs=train_label_set.unlabeled_idxs)
            else:
                train_unlabel_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train'], True), unlabeled_idxs=train_label_set.unlabeled_idxs)
                test_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=datasets.dict_transform['cifar_test'], unlabeled_idxs=train_label_set.unlabeled_idxs)
        elif args.dataset == 'cifar100-c':
            train_raw = np.load(f'./datasets/cifar/CIFAR-100-C/{args.degrade_choice}.npy')
            train_unlabel_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train'], True), unlabeled_idxs=train_label_set.unlabeled_idxs)
            train_unlabel_set.data = train_raw[train_label_set.unlabeled_idxs]
            test_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=datasets.dict_transform['cifar_test'], unlabeled_idxs=train_label_set.unlabeled_idxs)
            test_set.data = train_raw[train_label_set.unlabeled_idxs]
        num_classes = 100
    elif args.dataset == 'visda':
        num_classes = 12
        train_label_set = VisDAOW(root='./datasets/visda/', task='T', download=True, transform=TransformTwice(datasets.dict_transform['cifar_train'], True), labeled_num=args.labeled_num)
        if args.covariate_shift:
            train_unlabel_set = VisDAOW(root='./datasets/visda/', task='V', download=True, transform=TransformTwice(datasets.dict_transform['cifar_train'], True), labeled_num=args.labeled_num)
            test_set = train_unlabel_set
    else:
        warnings.warn('Dataset is not listed')
        return

    labeled_len = len(train_label_set)
    unlabeled_len = len(train_unlabel_set)
    labeled_batch_size = int(args.batch_size * labeled_len / (labeled_len + unlabeled_len))

    # Initialize the splits
    train_label_loader = torch.utils.data.DataLoader(train_label_set, batch_size=labeled_batch_size, shuffle=True, num_workers=3, drop_last=True)
    train_unlabel_loader = torch.utils.data.DataLoader(train_unlabel_set, batch_size=args.batch_size - labeled_batch_size, shuffle=True, num_workers=3, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=1)

    # First network intialization: pretrain the RotNet network
    model = odac_models.resnet18(num_classes=num_classes)
    if args.pretrained:
        # model.load_state_dict(torch.load('./pretrained/resnet18_pretrained.pth'), strict=False)
        if args.dataset == 'cifar10':
            state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
        elif args.dataset == 'cifar100':
            state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
        
    # Freeze the earlier filters
    for name, param in model.named_parameters():
        if 'fc' not in name and 'layer4' not in name:
            param.requires_grad = False
    # Set the optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-1, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    tf_writer = SummaryWriter(log_dir=args.savedir)

    for epoch in tqdm.tqdm(range(args.epochs), total=args.epochs):
        mean_uncert, overall_acc, seen_acc, unseen_acc = test(args, model, args.labeled_num, device, test_loader, epoch, tf_writer)
        train(args, model, device, train_label_loader, train_unlabel_loader, optimizer, mean_uncert, epoch, tf_writer)
        scheduler.step()
        if epoch == (args.epochs - 1):
            acc = {'overall_acc': overall_acc, 'seen_acc': seen_acc, 'unseen_acc': unseen_acc}
            with open('results.txt', 'a') as f:
                f.write(json.dumps(str(args.dataset)+str(args.covariate_shift)+str(args.degrade_level)+str(args.degrade_choice)+str(args.pretrained)+str(args.epochs)))
                f.write('\n')
                f.write(json.dumps(str(acc)))
                f.write('\n')
    
    # save model checkpoints
    checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(args.epochs)
    save_checkpoint({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=False, filename=os.path.join(tf_writer.log_dir, checkpoint_name))


if __name__ == '__main__':
    main()