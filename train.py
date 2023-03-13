import os
from re import A
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # [0, 1, 2, 3]
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import os
import random
import time
import torch
import torch.nn 
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import tensorboard_logger as tb_logger
import numpy as np
from model import ProtPLDA
from resnet import *
from utils.utils_algo import *
from utils.utils_loss import partial_loss, SupConLoss, Consi_entropy_loss, Non_candi_loss
import datetime
from utils.mnist import load_mnist
from utils.synthdigits import load_synthdigits
from utils.office31 import load_office31,load_officehome
from utils.visda17 import load_visda17
from utils.logger import CompleteLogger

torch.set_printoptions(precision=2, sci_mode=False)

parser = argparse.ArgumentParser(description='PyTorch implementation of Partial Label Domain Adaptation Contrastive Learning (PLDA)')
parser.add_argument('--dataset', default='office31_a_w', type=str, help='source2target')
parser.add_argument('--exp-dir', default='experiment/PLDA', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=['resnet18', 'resnet50', 'CNN', 'resnet34', 'resnet101'],
                    help='network architecture')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, 
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-lr_decay_epochs', type=str, default='200',
                    help='where to decay lr, can be a list')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU id to use.')
parser.add_argument('--num-class', default=10, type=int,
                    help='number of class')
parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension') # em
parser.add_argument('--moco_queue', default=8192, type=int, 
                    help='queue size; number of negative samples')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='momentum for updating momentum encoder')
parser.add_argument('--proto_m', default=0.99, type=float,
                    help='momentum for computing the momving average of prototypes')
parser.add_argument('--loss_weight', default=0.5, type=float,
                    help='contrastive loss weight')
parser.add_argument('--conf_ema_range', default='0.95,0.8', type=str,
                    help='pseudo target updating coefficient (phi)')
parser.add_argument('--prot_start', default=1, type=int, 
                    help = 'Start Prototype Updating')
parser.add_argument('--partial_rate', default=0.0, type=float, 
                    help='ambiguity level (q)')
parser.add_argument('--proto_threshold', default=0.99, type=float, 
                    help='Filtering threshold for updating prototype')
parser.add_argument('--filter_pro_init', default=0.5, type=float)
parser.add_argument('--filter_u_init', default=0.5, type=float)
parser.add_argument('--weight_cont', default=0.2, type=float)
parser.add_argument('--weight_selftrain', default=1.0, type=float)
parser.add_argument('--weight_neg', default=1.0, type=float)

def main():
    args = parser.parse_args()
    args.conf_ema_range = [float(item) for item in args.conf_ema_range.split(',')]
    args.gpu = [int(item) for item in args.gpu.split(',')]
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    
    model_path = 'ds_{ds}_pr_{pr}_lr_{lr}_Bsize{bs}_ep_{ep}_ps_{ps}arch_{arch}_filterinit{filter}_Date_{date}_contwei{contrastiveloss}_selfwei{selftrainloss}_negwei{negloss}_(no_Dis)_'.format(
                                            ds=args.dataset,
                                            pr=args.partial_rate,
                                            lr=args.lr,
                                            bs=args.batch_size,
                                            ep=args.epochs,
                                            ps=args.prot_start,
                                            #lw=args.loss_weight,
                                            #pm=args.proto_m,
                                            arch=args.arch,
                                            #seed=args.seed,
                                            filter=args.filter_pro_init,
                                            date=datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'),
                                            contrastiveloss=args.weight_cont,
                                            selftrainloss=args.weight_selftrain,
                                            negloss=args.weight_neg
                                            )
    args.exp_dir = os.path.join(args.exp_dir, model_path)
    record_log = CompleteLogger(args.exp_dir)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir) # experiment/PLDA/(check points)
    ngpus_per_node = torch.cuda.device_count() # 2
    main_worker(args.gpu, ngpus_per_node, args,record_log) 

def main_worker(gpu, ngpus_per_node, args, record_log):
    print(args)
    cudnn.benchmark = False
    args.gpu = gpu
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # create model
    print("=== > creating model '{}'".format(args.arch))
    if args.arch == 'CNN':
        Model = CNN_Digits
    else:
        Model = SupConResNet

    model = ProtPLDA(args, Model).cuda()
    # set optimiyzer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    if args.dataset == 'm2s': # mnist → svhn 
        partial_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/DACO/data/Partial_labels/mnist_{}.json'.format(args.partial_rate)
        source_loader, source_givenY = load_mnist(partial_rate=args.partial_rate, batch_size=args.batch_size, domain='source', partial_file=partial_file)
        target_loader, target_truthY, tar_ori_dataset  = load_synthdigits(partial_rate=args.partial_rate, batch_size=args.batch_size, domain='target')
        test_loader = load_synthdigits(partial_rate=args.partial_rate, batch_size=args.batch_size, domain='test') # test data
        test_loader_s = load_mnist(partial_rate=args.partial_rate, batch_size=args.batch_size, domain='test')
    elif args.dataset == 's2m':
        partial_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/DACO/data/Partial_labels/synthdigits_{}.json'.format(args.partial_rate)
        source_loader, source_givenY = load_synthdigits(partial_rate=args.partial_rate, batch_size=args.batch_size, domain='source', partial_file=partial_file)
        target_loader, target_truthY, tar_ori_dataset = load_mnist(partial_rate=args.partial_rate, batch_size=args.batch_size, domain='target')
        test_loader = load_mnist(partial_rate=args.partial_rate, batch_size=args.batch_size, domain='test') # test data
        test_loader_s = load_synthdigits(partial_rate=args.partial_rate, batch_size=args.batch_size, domain='test') # test data
    elif args.dataset[0:8] == 'office31':
        s_name, t_name = get_domain(args)
        partial_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/DACO/data/Partial_labels/{}_{}.json'.format(s_name,args.partial_rate)
        source_loader, source_givenY = load_office31(partial_rate=args.partial_rate, batch_size=args.batch_size, domain='source', name=s_name,partial_file=partial_file)
        target_loader, target_truthY, tar_ori_dataset = load_office31(partial_rate=args.partial_rate, batch_size=args.batch_size, domain='target', name=t_name, partial_file=partial_file)
        test_loader = load_office31(partial_rate=args.partial_rate, batch_size=args.batch_size, domain='test', name=t_name)
        test_loader_s = load_office31(partial_rate=args.partial_rate, batch_size=args.batch_size, domain='test', name=s_name)
    elif args.dataset[0:10] == 'officehome':
        s_name, t_name = get_domain(args)
        partial_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/DACO/data/Partial_labels/{}_{}.json'.format(s_name,args.partial_rate)
        source_loader, source_givenY = load_officehome(partial_rate=args.partial_rate, batch_size=args.batch_size, domain='source', name=s_name,partial_file=partial_file)
        target_loader, target_truthY, tar_ori_dataset = load_officehome(partial_rate=args.partial_rate, batch_size=args.batch_size, domain='target', name=t_name, partial_file=partial_file)
        test_loader = load_officehome(partial_rate=args.partial_rate, batch_size=args.batch_size, domain='test', name=t_name)
        test_loader_s = load_officehome(partial_rate=args.partial_rate, batch_size=args.batch_size, domain='test', name=s_name)
    elif args.dataset == 'visda2017':
        partial_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/DACO/data/Partial_labels/VisDa_S_{}.json'.format(args.partial_rate)
        s_name, t_name = get_domain(args)
        source_loader, source_givenY = load_visda17(args,
                                                task=s_name,
                                                domain='source',
                                                partial_file=partial_file)
        target_loader, tar_ori_dataset = load_visda17(args,task=t_name,domain='target')
        test_loader = load_visda17(args,task=t_name,domain='test')
        #test_loader_s = load_visda17(args,task=s_name,domain='test')
    else:
        raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")
    
    # this train loader is the partial label training loader
    print('Calculating uniform targets...')
    tempY = source_givenY.sum(dim=1).unsqueeze(1).repeat(1, source_givenY.shape[1]) # .repeat(1, n) 行方向不变，沿列方向重复n次
    confidence = source_givenY.float()/tempY # initialization
    confidence = confidence.cuda()
    # calculate confidence
    
    loss_fn = partial_loss(confidence) # 偏标记损失 confidence = pseudo_targets
    loss_cont_fn = SupConLoss().cuda() # in-domain
    loss_tar = torch.nn.CrossEntropyLoss().cuda()
    # define object function
    logger = tb_logger.Logger(logdir=os.path.join(args.exp_dir,'tensorboard'), flush_secs=2)
    

    print('\n === Start Training === \n')

    best_acc = 0.
    mmc = 0.
    # new pseudo-labels module
    prototype_source = None
    prototype_target = None

    for epoch in range(args.start_epoch, args.epochs):
        start_upd_prot = epoch>=args.prot_start # 40
        adjust_learning_rate(args, optimizer, epoch)
        if epoch >= args.prot_start - 1: # filter reliable pseudo-labels for target domain
            target_loader, target_u_loader = Generate_tar_pse(model, tar_ori_dataset, args, epoch,prototype_source,prototype_target)
        else:
            target_u_loader = None
        prototype_source, prototype_target = train(source_loader, target_loader, model, loss_fn, loss_cont_fn, loss_tar, optimizer, epoch, args, logger, start_upd_prot, target_u_loader=target_u_loader)
        loss_fn.set_conf_ema_m(epoch, args) # ?
        # reset phi
        #acc_test_s = test(model, test_loader_s, args, epoch, logger, 'source')
        Original_t_features, acc_test,everyclass_acc = test(model, test_loader, args, epoch, logger, 'target')
        #logger.log_value('Test Accuracy(Source)', acc_test_s, epoch)
        logger.log_value('Test Accuracy(Target)', acc_test, epoch) # test acc target
        mmc = loss_fn.confidence.max(dim=1)[0].mean() # ?
        
        if acc_test > best_acc:
            best_acc = acc_test
            is_best = True
            np.save('/media/sobremesa/E/DACO_Best_NanQI/DACO/experiment/New/feature_ablation/featuresT(A2W)_{}(nossp).npy'.format(args.partial_rate), Original_t_features.cpu().numpy())

        with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f: # a+ 追加且可读可写
            f.write('Epoch {}: Target Acc {}, Best Target Acc {}, (lr {}), EveryClassAcc {}\n'.format(epoch
                , acc_test, best_acc, optimizer.param_groups[0]['lr'], everyclass_acc))

    record_log.close()

def train(source_loader, target_loader, model, loss_fn, loss_cont_fn, loss_tar, optimizer, epoch, args, tb_logger, start_upd_prot=False, target_u_loader=None):
    batch_time = AverageMeter('Time', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    acc_cls_t = AverageMeter('Acc_tar@Cls_top1', ':2.2f')
    acc_proto = AverageMeter('Acc@Proto(Source)', ':2.2f')
    loss_cls_log = AverageMeter('Loss@Cls', ':2.2f')
    loss_cont_log = AverageMeter('Loss@Cont', ':2.2f')
    loss_cls_neg_log = AverageMeter('Loss@Cls(Neg)', ':2.2f')

    num_iter = min(1024, len(source_loader)) 
    #num_iter = len(source_loader)
    progress = ProgressMeter(
        num_iter,
        [batch_time, acc_cls, acc_cls_t, acc_proto, loss_cls_log, loss_cont_log, loss_cls_neg_log],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    #num_iter = min(len(source_loader), len(target_loader))
    #len(target_loader)#len(target_loader)
    # iter dataloader
    iter_source = iter(source_loader)
    iter_target = iter(target_loader)
    sou_proto_que = None
    tar_proto_que = None
    ## validation ##
    for i in range(num_iter):
        try:
            batch_sou = iter_source.next()
        except:
            iter_source = iter(source_loader)
            batch_sou = iter_source.next()
        try:
            batch_tar = iter_target.next()
        except:
            iter_target = iter(target_loader)
            batch_tar = iter_target.next()
        
        # image_aug_domain
        X_w_s, X_s_s, p_labels_s, true_labels_s, index_s = batch_sou
        X_w_t, X_s_t, p_labels_t, true_labels_t, index_t = batch_tar \
        

        # source
        X_w_s, X_s_s, p_labels_s, true_labels_s = X_w_s.cuda(), X_s_s.cuda(), p_labels_s.cuda(), true_labels_s.detach().cuda()
        # labeled tar
        X_w_t, X_s_t, p_labels_t, true_labels_t = X_w_t.cuda(), X_s_t.cuda(), p_labels_t.cuda(), true_labels_t.cuda()
        
        cls_out, features_cont, features_cont_tar, pseudo_score_cont, score_prot, sou_proto_que, tar_proto_que, output_t_strong \
            = model(X_w_s, X_s_s, p_labels_s,  X_w_t, X_s_t, p_labels_t, true_labels_t, args, epoch=epoch)

        batch_size = cls_out.shape[0]
        # source pseudo
        pseudo_target_max, pseudo_target_cont = torch.max(pseudo_score_cont, dim=1)
        pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1)

        p_labels_t_cont = p_labels_t.contiguous().view(-1, 1)

        if start_upd_prot:
            # for validate disambiguation
            loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index_s, batchY=p_labels_s)
            #pass
        #------------------------------------------- loss --------------------------------------------#
        loss_cls = loss_fn(cls_out, index_s)
        # ablation !!
        loss_neg = -torch.mean(torch.sum(torch.log(1.0000001 - F.softmax(cls_out, dim=1)) * (1 - p_labels_s), dim=1))
        
        if start_upd_prot:
            class_index = torch.arange(args.num_class).cuda()
            mask_s_s = torch.eq(pseudo_target_cont[:batch_size], class_index).float().cuda()
            loss_cont_ss = loss_cont_fn(features=features_cont, mask=mask_s_s, batch_size=batch_size, prototype_que=sou_proto_que,if_proto=True) # S2S
            mask_s_t = torch.eq(pseudo_target_cont[:batch_size], class_index)
            loss_cont_st = loss_cont_fn(features=features_cont, mask=mask_s_t, batch_size=batch_size, prototype_que=tar_proto_que,if_proto=True) # S2S
            # target unlabeled data for clustering
            mask_t_t = torch.eq(p_labels_t_cont, class_index).float().cuda()
            loss_cont_tt = loss_cont_fn(features=features_cont_tar, mask=mask_t_t, batch_size=batch_size, prototype_que=tar_proto_que,if_proto=True)
            mask_t_s = torch.eq(p_labels_t_cont, class_index).float().cuda()
            loss_cont_ts = loss_cont_fn(features=features_cont_tar, mask=mask_t_s, batch_size=batch_size, prototype_que=sou_proto_que,if_proto=True)

            #loss_cont = loss_cont_ss + loss_cont_tt + 0.2 * (loss_cont_st + loss_cont_ts) # 最本质的两个核心loss 是 loss_ss + loss_tt

            loss_cont = loss_cont_ss + loss_cont_tt + (loss_cont_st + loss_cont_ts) # 最本质的两个核心loss 是 loss_ss + loss_tt

            loss_cls_tar = loss_tar(output_t_strong, p_labels_t)
            ### 
            loss = loss_cls + args.weight_neg * loss_neg + args.weight_cont * loss_cont + args.weight_selftrain * loss_cls_tar

        else: # stage 1
            mask_s_s = None
            loss_cont = loss_cont_fn(features=features_cont, mask=mask_s_s, batch_size=batch_size)
            # Warmup using MoCo     
            loss = loss_cls + args.weight_neg * loss_neg + loss_cont

        
        loss_cls_log.update(loss_cls.item())
        loss_cont_log.update(loss_cont.data.item())
        loss_cls_neg_log.update(loss_neg.data.item())
        

        # log accuracy
        acc,_ = accuracy(cls_out, true_labels_s) # accuracy on source domain
        acc_cls.update(acc[0].item())
        #print('Acc source_2:', acc)
        acc,_ = accuracy(score_prot, true_labels_s) # accuracy on proto
        acc_proto.update(acc[0].item())

        if start_upd_prot:
            acc,acc_predict = accuracy(F.one_hot(p_labels_t,num_classes=args.num_class).float(), true_labels_t, topk=(1, ),if_everyclass=True,args=args) # accuracy on filter target domain
            acc_cls_t.update(acc[0].item())
            with open(os.path.join(args.exp_dir, 'Everyclass_target(predict).log'), 'a+') as f: # a+ 追加且可读可写
                f.write('Epoch:{}, iter:{}, predict_acc:{}\n \n'.format(epoch,i,acc_predict))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)

    print("\n")

    tb_logger.log_value('Train Acc', acc_cls.avg, epoch) # source acc during training
    tb_logger.log_value('Target Acc', acc_cls_t.avg, epoch) # target acc during training
    tb_logger.log_value('Prototype Acc Source', acc_proto.avg, epoch) # source pseudo acc 
    #tb_logger.log_value('Prototype Acc Target', acc_proto_t.avg, epoch)
    tb_logger.log_value('Classification Loss', loss_cls_log.avg, epoch)
    tb_logger.log_value('Contrastive Loss', loss_cont_log.avg, epoch)
    #return tar_proto_que
    if epoch >= args.prot_start - 1:
        return sou_proto_que, tar_proto_que
    else:
        return None, None


def test(model, test_loader, args, epoch, tb_logger, domain=None):
    with torch.no_grad():
        print('======= Evaluation========\n')       
        model.eval()    
        Test_Acc_top1 = AverageMeter("TestAcc_top1")
        class_total = list(0. for i in range(args.num_class)) #用来计算每个类别的测试样本总数
        class_corr = list(0. for i in range(args.num_class)) #用来计算每个类别计算正确的样本个数
        Original_t_features = None
        first_Test = True
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            features_batch, outputs,_ = model(img_t_q=images, args=args, eval_only=True)
            ## For T-SNE
            if first_Test:
                Original_t_features = features_batch
                first_Test = False
            else:
                Original_t_features = torch.cat((Original_t_features,features_batch),0)    

            acc,class_corr_num = accuracy(outputs, labels, topk=(1,),if_everyclass=False,args=args,final_test=True)
            for label in labels:
                class_total[int(label)]+=1. 
            for label in range(args.num_class):
                class_corr[label]+=class_corr_num[label]
            Test_Acc_top1.update(acc[0].item())
        acc_class = list(0. for i in range(args.num_class))
        # 计算最终的每个类别准确率
        for label in range(args.num_class):
            acc_class[label] = 100 * (class_corr[label] / class_total[label])
        # average across all processes
        acc_tensors = torch.Tensor([Test_Acc_top1.avg]).cuda()
        acc_tensors /= 1
        print('=== The top 1 Accuracy(%s) is %.2f%% ==='%(domain, acc_tensors[0]))
        if args.gpu ==0:
            tb_logger.log_value('Top1 Acc', acc_tensors[0], epoch)    
    return Original_t_features,acc_tensors[0],acc_class
    #return acc
           
def save_checkpoint(state, is_best, best_file_name='model_best.pth.tar'): #filename='checkpoint.pth.tar',
    torch.save(state, best_file_name)
    # if is_best:
    #     shutil.copyfile(filename, best_file_name)

if __name__ == '__main__':
    main()
