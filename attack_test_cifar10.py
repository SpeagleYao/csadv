from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from models import PreActResNet18
from tqdm import tqdm
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
# TODO:
parser.add_argument('--model-path',
                    default='./cp_cifar10/res18_natural.pth',
                    help='model for white-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    f2t_nat = np.logical_and(out.data.max(1)[1].cpu().numpy() == 3, y.data.cpu().numpy() == 5).sum()
    t2f_nat = np.logical_and(out.data.max(1)[1].cpu().numpy() == 5, y.data.cpu().numpy() == 3).sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    out_pgd = model(X_pgd)
    err_pgd = (out_pgd.data.max(1)[1] != y.data).float().sum()
    f2t_pgd = np.logical_and(out_pgd.data.cpu().numpy().max(1)[1] == 3, y.data.cpu().numpy() == 5).sum()
    t2f_pgd = np.logical_and(out_pgd.data.cpu().numpy().max(1)[1] == 5, y.data.cpu().numpy() == 3).sum()
    # print('err pgd (white-box): ', err_pgd)
    return err, err_pgd, f2t_nat, t2f_nat, f2t_pgd, t2f_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    t2f_nat_total = 0
    f2t_nat_total = 0
    t2f_rob_total = 0
    f2t_rob_total = 0

    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust, f2t_nat_err, t2f_nat_err, f2t_rob_err, t2f_rob_err = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
        f2t_nat_total += f2t_nat_err
        t2f_nat_total += t2f_nat_err
        f2t_rob_total += f2t_rob_err
        t2f_rob_total += t2f_rob_err
    print('natural_acc:\t', 100 - natural_err_total.cpu().numpy()/100, '%')
    print('robust _acc:\t', 100 - robust_err_total.cpu().numpy()/100, '%')
    print('cat to dog nat:\t', 100 - f2t_nat_total/10, '%')
    print('dog to cat nat:\t', 100 - t2f_nat_total/10, '%')
    print('cat to dog rob:\t', 100 - f2t_rob_total/10, '%')
    print('dog to cat rob:\t', 100 - t2f_rob_total/10, '%')


def main():

    print('pgd white-box attack')
    model = PreActResNet18().to(device)
    model.load_state_dict(torch.load(args.model_path))

    eval_adv_test_whitebox(model, device, test_loader)


if __name__ == '__main__':
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    main()
