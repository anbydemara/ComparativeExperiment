import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from tensorboardX import SummaryWriter
from torch import optim
import matplotlib.pyplot as plt
from con_losses import SupConLoss, CosineContrastiveLoss
from datasets import get_dataset, HyperX, HyperV
from network import discriminator
from network import generator
from utils_HSI import sample_gt, metrics, seed_worker
import get_cls_map

parser = argparse.ArgumentParser(description='PyTorch SDEnet')
parser.add_argument('--save_path', type=str, default='./results/')
# parser.add_argument('--data_path', type=str, default='./YC/')
#
# parser.add_argument('--source_name', type=str, default='GFYC',
#                     help='the name of the source dir')
# parser.add_argument('--target_name', type=str, default='ZYYC',
#                     help='the name of the test dir')
parser.add_argument('--data_path', type=str, default='./Pavia/')
parser.add_argument('--source_name', type=str, default='paviaU',
                    help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='paviaC',
                    help='the name of the test dir')
# parser.add_argument('--data_path', type=str, default='./Houston/')
#
# parser.add_argument('--source_name', type=str, default='Houston13',
#                     help='the name of the source dir')
# parser.add_argument('--target_name', type=str, default='Houston18',
#                     help='the name of the test dir')
parser.add_argument('--gpu', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")

parser.add_argument('--dim1', type=int, default=8)
parser.add_argument('--dim2', type=int, default=16)

group_train = parser.add_argument_group('Training')
group_train.add_argument('--patch_size', type=int, default=13,
                         help="Size of the spatial neighbourhood (optional, if ""absent will be set by the model)")

group_train.add_argument('--lr', type=float, default=0.001,
                         help="Learning rate, set by the model if not specified.")
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.5)')
group_train.add_argument('--batch_size', type=int, default=256,
                         help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--pro_dim', type=int, default=128)
group_train.add_argument('--test_stride', type=int, default=1,
                         help="Sliding window step stride during inference (default = 1)")
parser.add_argument('--seed', type=int, default=233,
                    help='random seed ')
parser.add_argument('--l2_decay', type=float, default=1e-4,
                    help='the L2  weight decay')
parser.add_argument('--num_epoch', type=int, default=500,
                    help='the number of epoch')
parser.add_argument('--training_sample_ratio', type=float, default=0.8,
                    help='training sample ratio')
parser.add_argument('--re_ratio', type=int, default=1,  # PaviaU-1 Houston13-5，图像扩充倍数
                    help='multiple of of data augmentation')
parser.add_argument('--max_epoch', type=int, default=400)
parser.add_argument('--log_interval', type=int, default=40)
parser.add_argument('--lambda_1', type=float, default=1.0)
parser.add_argument('--lambda_2', type=float, default=1.0)
parser.add_argument('--lr_scheduler', type=str, default='none')

group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=False,
                      help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=False,
                      help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                      help="Random mixes between spectra")
args = parser.parse_args()


def evaluate(net, val_loader, gpu, tgt=False):
    ps = []
    ys = []
    for i, (x1, y1) in enumerate(val_loader):
        y1 = y1 - 1
        with torch.no_grad():
            x1 = x1.to(gpu)
            p1 = net(x1)
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys == ps) * 100
    if tgt:
        results = metrics(ps, ys, n_classes=ys.max().astype(int) + 1)  # metrics 和 show_results 均是可直接使用的HSI计算工具
        print(results['Confusion_matrix'], '\n', 'TPR:', np.round(results['TPR'] * 100, 2), '\n', 'OA:',
              results['Accuracy'], 'Kappa:', results["Kappa"])
    return acc


def evaluate_tgt(cls_net, gpu, loader, modelpath):
    saved_weight = torch.load(modelpath)
    cls_net.load_state_dict(saved_weight['Discriminator'])
    cls_net.eval()
    teacc = evaluate(cls_net, loader, gpu, tgt=True)
    return teacc


def experiment():
    settings = locals().copy()
    print(settings)
    hyperparams = vars(args)
    print(hyperparams)
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    root = os.path.join(args.save_path, args.source_name + 'to' + args.target_name)
    log_dir = os.path.join(root, str(args.lr) + '_dim' + str(args.pro_dim) +
                           '_pt' + str(args.patch_size) + '_bs' + str(args.batch_size) + '_' + time_str)
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    df = pd.DataFrame([args])
    df.to_csv(os.path.join(log_dir, 'params.txt'))

    seed_worker(args.seed)
    img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name,
                                                                                        args.data_path)
    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                                                        args.data_path)
    sample_num_src = len(np.nonzero(gt_src)[0])
    sample_num_tar = len(np.nonzero(gt_tar)[0])

    tmp = args.training_sample_ratio * args.re_ratio * sample_num_src / sample_num_tar
    num_classes = gt_src.max().astype(int)
    N_BANDS = img_src.shape[-1]
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': args.gpu, 'center_pixel': None, 'supervision': 'full'})

    r = int(hyperparams['patch_size'] / 2) + 1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    train_gt_src, val_gt_src, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')  # 划分训练集和验证集保持了类别比例
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src
    val_gt_src_con = val_gt_src
    if tmp < 1:  # 如果预计增广后的训练样本数量少于测试样本数量，才真的对训练样本+验证样本增广
        for i in range(args.re_ratio - 1):
            img_src_con = np.concatenate((img_src_con, img_src))
            train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))
            val_gt_src_con = np.concatenate((val_gt_src_con, val_gt_src))

    hyperparams_train = hyperparams.copy()
    g = torch.Generator()
    g.manual_seed(args.seed)
    vision_dataset = HyperV(img_tar, gt_tar, **hyperparams)
    vision_loader = test_loader = data.DataLoader(vision_dataset,
                                                  pin_memory=True,
                                                  worker_init_fn=seed_worker,
                                                  generator=g,
                                                  batch_size=hyperparams['batch_size'])
    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   pin_memory=True,
                                   worker_init_fn=seed_worker,
                                   generator=g,
                                   shuffle=True, )
    val_dataset = HyperX(img_src_con, val_gt_src_con, **hyperparams)
    val_loader = data.DataLoader(val_dataset,
                                 pin_memory=True,
                                 batch_size=hyperparams['batch_size'])
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset,
                                  pin_memory=True,
                                  worker_init_fn=seed_worker,
                                  generator=g,
                                  batch_size=hyperparams['batch_size'])
    imsize = [hyperparams['patch_size'], hyperparams['patch_size']]

    # =================鉴别器模型配置=================
    D_net = discriminator.Discriminator(inchannel=N_BANDS, outchannel=args.pro_dim, num_classes=num_classes,
                                        patch_size=hyperparams['patch_size']).to(args.gpu)
    # =================生成器模型配置=================
    G_net = generator.Generator_3DCNN_SupCompress_pca(imdim=N_BANDS, imsize=imsize, dim1=args.dim1, dim2=args.dim2,
                                                      device=args.gpu).to(args.gpu)
    # =================模型LOSS与训练优化配置=================
    D_opt = optim.Adam(D_net.parameters(), lr=args.lr)
    G_opt = optim.Adam(G_net.parameters(), lr=args.lr)
    cls_criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(device=args.gpu)

    alpha_values = []

    # =================开始训练=================
    best_acc = 0
    taracc, taracc_list = 0, []
    for epoch in range(1, args.max_epoch + 1):

        t1 = time.time()
        loss_list = []
        D_net.train()

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(args.gpu), y.to(args.gpu)
            y = y - 1
            with torch.no_grad():  # 将模型前向传播的代码放到with torch.no_grad()下，就能使pytorch不生成计算图，从而节省不少显存
                x_ED = G_net(x)
            rand = torch.nn.init.uniform_(torch.empty(len(x), 1, 1, 1)).to(args.gpu)  # Uniform distribution
            x_ID = rand * x + (1 - rand) * x_ED

            x_tgt = G_net(x)

            p_SD, z_SD = D_net(x, mode='train')
            p_ED, z_ED = D_net(x_ED, mode='train')
            p_ID, z_ID = D_net(x_ID, mode='train')
            zsrc = torch.cat([z_SD.unsqueeze(1), z_ID.unsqueeze(1), z_ED.unsqueeze(1)], dim=1)
            src_cls_loss = cls_criterion(p_SD, y.long()) + cls_criterion(p_ED, y.long()) + cls_criterion(p_ID, y.long())
            p_tgt, z_tgt = D_net(x_tgt, mode='train')
            tgt_cls_loss = cls_criterion(p_tgt, y.long())  # 辅助损失
            # tgt_cls_loss = cls_criterion(p_ED, y.long())  #消融

            zall = torch.cat([z_tgt.unsqueeze(1), zsrc], dim=1)
            con_loss = con_criterion(zall, y, adv=False)
            loss1 = src_cls_loss + args.lambda_1 * con_loss
            # loss1 = src_cls_loss #消融
            D_opt.zero_grad()  # 先只优化D_net
            loss1.backward(retain_graph=True)  # 不释放计算图，对Loss2进行计算时，梯度是累加的

            num_adv = y.unique().size()
            zsrc_con = torch.cat([z_tgt.unsqueeze(1), z_ED.unsqueeze(1).detach()],
                                 dim=1)
            # zsrc_con = torch.cat([z_ID.unsqueeze(1), z_ED.unsqueeze(1).detach()],
            #                      dim=1)   #消融
            con_loss_adv = 0
            idx_1 = np.random.randint(0, zsrc_con.size(1))
            for i, id in enumerate(y.unique()):
                mask = y == y.unique()[i]
                z_SD_i, zsrc_i = z_SD[mask], zsrc_con[mask]
                y_i = torch.cat([torch.zeros(z_SD_i.shape[0]), torch.ones(z_SD_i.shape[0])])  # 打上新的真伪标签，真-0，伪-1
                zall = torch.cat([z_SD_i.unsqueeze(1).detach(), zsrc_i[:, idx_1:idx_1 + 1]],
                                 dim=0)
                if y_i.size()[0] > 2:
                    con_loss_adv += con_criterion(zall, y_i)


            con_loss_adv = con_loss_adv / y.unique().shape[0]  # 计算平均每个类别的真伪对比损失

            loss2 = tgt_cls_loss + args.lambda_2 * con_loss_adv
            # loss2 = tgt_cls_loss #消融


            G_opt.zero_grad()
            loss2.backward()
            D_opt.step()


            G_opt.step()

        D_net.eval()
        teacc = evaluate(D_net, val_loader, args.gpu)
        if best_acc < teacc:
            best_acc = teacc
            # 保存参数的时候不要同时运行另一个调用该参数的程序，否则不会被成功保存
            # 只保存训练模型的可学习参数
            torch.save({'Discriminator': D_net.state_dict()}, os.path.join(log_dir, f'best.pkl'))
            torch.save({'Generator': G_net.state_dict()}, os.path.join(log_dir, f'best_G.pkl'))
        t2 = time.time()

        print(
            f'epoch {epoch}, train {len(train_loader.dataset)}, time {t2 - t1:.2f}, src_cls {src_cls_loss:.4f} tgt_cls {tgt_cls_loss:.4f} conIE {con_loss:.4f}, con_adv {con_loss_adv:.4f}// val {len(val_loader.dataset)}, teacc {teacc:2.2f}')

        if epoch % args.log_interval == 0:
            pklpath = f'{log_dir}/best.pkl'
            taracc = evaluate_tgt(D_net, args.gpu, test_loader, pklpath)
            taracc_list.append(round(taracc, 2))
            print(f'load pth, target sample number {len(test_loader.dataset)}, max taracc {max(taracc_list):2.2f}')
            get_cls_map.get_cls_map(D_net, args.gpu, vision_loader, gt_tar, epoch)



if __name__ == '__main__':
    experiment()
