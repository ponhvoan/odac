import argparse
import torch 
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from utils import accuracy, degrade_data, TransformTwice
import torch.nn.functional as F
import numpy as np
import os
import glob
import csv

import open_world_cifar as datasets
import open_world_imagenet as im_datasets
import odac_models
from torchvision import transforms

import sys
sys.path.insert(0, '/home/voan/opencon')
from opencon_models.OpenSupCon import OpenSupCon
sys.path.remove('/home/voan/opencon')

sys.path.insert(0, '/home/voan/SimCLR')
from models.resnet_simclr import ResNetSimCLR
sys.path.remove('/home/voan/SimCLR')

class NoneTransform(object):
    """ Does nothing to the image, to be used instead of None
    
    Args:
        image in, image out, nothing is done
    """
    def __call__(self, image):       
        return image

transform_visda_target = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(112),
                transforms.ToTensor(),
            ])

transform_imagenet = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

def get_test_loader(args):

    if args.degrade_choice != 'none' and args.dataset != 'visda':
        degrade = degrade_data(degrade_choice=args.degrade_choice, degrade_level=args.degrade_level)
        transform_test = degrade
    else:
        transform_test = NoneTransform()
    if args.dataset == 'cifar10':
        train_label_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']))
        test_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=transforms.Compose([datasets.dict_transform['cifar_test'], transform_test]), unlabeled_idxs=train_label_set.unlabeled_idxs)
    elif args.dataset == 'cifar100':
        train_label_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']))
        test_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=transforms.Compose([datasets.dict_transform['cifar_test'], transform_test]), unlabeled_idxs=train_label_set.unlabeled_idxs)
    elif args.dataset == 'visda':        
        test_set = im_datasets.VisDA(root='./datasets/visda/', anno_file=f'./data/visda_unlabel_{args.labeled_num}_{args.labeled_ratio:.1f}.txt', source='t', transform=transform_visda_target)
    elif args.dataset == 'imagenet100':
        test_set = im_datasets.ImageNetDataset(root='/home/hirokiwaida/data/imagenet/train/', anno_file=f'/home/voan/orca/data/ImageNet100_unlabel_{args.labeled_num}_{args.labeled_ratio:.1f}.txt', transform=transforms.Compose([transform_imagenet, transform_test]))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=1)

    return test_loader

def cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    w = w[y_true.min():, y_true.min():]
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    row_col_pair = [(row_ind[i], col_ind[i]) for i in range(len(row_ind))] 

    return w[row_ind, col_ind].sum() / y_pred.size , row_col_pair

def test(model, test_loader, args):
    
    model.eval()
    preds = np.array([])
    targets = np.array([])
    # confs = np.array([])
    if args.method == 'simclr':
        outputs = np.empty([0,out_dim])
    else:
        outputs = np.empty([0,args.num_classes])
    with torch.no_grad():
        for batch_idx, (x, label, _) in enumerate(test_loader):
            x, label = x.to(args.device), label.to(args.device)

            if args.method == 'opencon':
                ret_dict = model.forward_cifar(x, None, label, evalmode=True)
                pred = ret_dict['label_pseudo']
                # conf = ret_dict['conf']
                targets = np.append(targets, label.cpu().numpy())
                preds = np.append(preds, pred.cpu().numpy())
                # confs = np.append(confs, conf.cpu().numpy())
            else:
                output, _ = model(x)
                prob = F.softmax(output, dim=1)
                conf, pred = prob.max(1)
                targets = np.append(targets, label.cpu().numpy())
                preds = np.append(preds, pred.cpu().numpy())
                # confs = np.append(confs, conf.cpu().numpy())
                outputs = np.vstack([outputs, output.cpu().numpy()])
    targets = targets.astype(int)
    preds = preds.astype(int)
    seen_mask = targets < args.labeled_num
    unseen_mask = ~seen_mask

    if args.method != 'simclr':
        seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
        unseen_acc, _ = cluster_acc(preds[unseen_mask], targets[unseen_mask])
        overall_acc = (seen_acc*sum(seen_mask) + unseen_acc*sum(unseen_mask))/len(targets)
        unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
        # mean_uncert = 1 - np.mean(confs)
    else:
        # find optimal assignment
        Kmean_overall = KMeans(n_clusters=args.num_classes)
        Kmean_overall.fit(outputs)
        y_pred = Kmean_overall.predict(outputs)
        overall_acc = accuracy_score(y_pred, targets)
        seen_acc = accuracy_score(y_pred[seen_mask], targets[seen_mask])
        unseen_acc = accuracy_score(y_pred[unseen_mask], targets[unseen_mask])
        # overall_acc, row_col_pair = cluster_acc(Kmean_overall.labels_, targets)
        # cluster_labels = Kmean_overall.labels_
        # for i in range(len(preds)):
        #     cluster_labels[i] = row_col_pair[cluster_labels[i]][1]

        # # accuracy
        # seen_acc = accuracy(cluster_labels[seen_mask], targets[seen_mask])
        # unseen_acc = accuracy(cluster_labels[unseen_mask], targets[unseen_mask])
        # overall_acc = (seen_acc*sum(seen_mask) + unseen_acc*sum(unseen_mask))/len(targets)

    return seen_acc, unseen_acc, overall_acc

def load_model(args):

    root = '/home/voan'
    if args.method == 'odac' or args.method == 'orca':
        model_path = os.path.join(root, f'orca/results_final/{args.dataset}/{args.degrade_choice}{str(args.degrade_level)}/{args.method}')
        model_path = os.path.join(model_path, '0*')
    elif args.method == 'simclr':
        model_path = os.path.join(root, f'SimCLR/runs/{args.dataset}/{args.degrade_choice}{str(args.degrade_level)}/noreg')
        model_path = os.path.join(model_path, '*')
    elif args.method == 'opencon':
        model_path = os.path.join(root, f'opencon/results/opencon-{args.dataset}/{args.degrade_choice}{str(args.degrade_level)}')

    if args.method != 'opencon':
        # model_path = sorted(glob.glob(model_path))[0]
        model_path_list = glob.glob(model_path)
        for path in model_path_list:
            if f'{str(args.gamma)}{str(args.labeled_num)}{str(args.labeled_ratio)}' in path:# and 'nocl' in path:
                model_path = path
        model_path = os.path.join(model_path, f'checkpoint_{args.epochs:04d}.pth.tar')
    else:
        if args.dataset == 'imagenet100':
            model_path = os.path.join(model_path, 'snapshot', '74.pth')
        else:
            model_path = os.path.join(model_path, 'snapshot', f'{args.epochs-1}.pth')
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']

    if args.method == 'orca' or args.method == 'odac':
        if 'cifar' in args.dataset:
            model = odac_models.resnet18(num_classes=args.num_classes)
        elif args.dataset == 'imagenet100' or args.dataset=='visda':
            model = odac_models.resnet50(num_classes=args.num_classes)
    elif args.method == 'opencon':
        if 'cifar' in args.dataset:
            model = OpenSupCon("RN18_simclr_CIFAR", args)
        elif args.dataset == 'imagenet100' or args.dataset== 'visda':
            model = OpenSupCon("RN50_simclr", args)
    elif args.method == 'simclr':
        global out_dim 
        out_dim = state_dict[next(reversed(state_dict))].shape[-1]
        if 'cifar' in args.dataset:
            model = ResNetSimCLR(base_model='resnet18_orca', out_dim=out_dim)
        elif args.dataset == 'imagenet100' or args.dataset == 'visda':
            model = ResNetSimCLR(base_model='resnet50_orca', out_dim=out_dim)
    
    model.load_state_dict(state_dict, strict=True)

    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation with baselines')
    parser.add_argument('--dataset', default='cifar100', help='dataset setting')
    parser.add_argument('--method', default='odac')
    parser.add_argument('--proto-num', default=100, type=int, help='prototype number for opencon')
    parser.add_argument('--momentum-proto', type=float, default=0.9)
    parser.add_argument('--name', type=str, default='opencon')
    parser.add_argument('--covariate-shift', default=True, action=argparse.BooleanOptionalAction, 
                    help='boolean flag for whether covariate shift is present')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--labeled-num', default=50, type=int)
    parser.add_argument('--labeled-ratio', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.4, type=float)
    parser.add_argument('--degrade-level', type=int, default=1, choices=[0,1,2,3])
    parser.add_argument('--degrade-choice', type=str, default='jitter', choices=['none', 'blur', 'jitter', 'elastic'],
                    help='choice of dataset shift')
    args = parser.parse_args()
    cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if cuda else "cpu")
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.proto_num = 10
    elif args.dataset == 'cifar100' or args.dataset=='imagenet100':
        args.num_classes = 100
        args.proto_num = 100
    elif args.dataset == 'visda':
        args.num_classes = 12 
        args.proto_num = 12
    
    model = load_model(args)    
    model = model.to(args.device)
    test_loader = get_test_loader(args)

    seen_acc, unseen_acc, overall_acc = test(model, test_loader, args)
    print(args.dataset, args.degrade_choice, args.degrade_level, args.method)
    print(seen_acc, unseen_acc, overall_acc)
    acc_dict = {f'{args.dataset}_{args.degrade_choice}{args.degrade_level}_{args.method}_{args.gamma}_{args.labeled_num}_{args.labeled_ratio}': [seen_acc, unseen_acc, overall_acc]}

    with open('eval_final.csv', 'a+') as out:
        writer = csv.writer(out)
        for key, value in acc_dict.items():
            writer.writerow([key] + value)