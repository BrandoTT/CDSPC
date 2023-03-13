from ast import arg
from tabnanny import verbose
from tkinter import Variable
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.utils_algo import *
from torch.autograd import Variable

class ProtPLDA(nn.Module):

    def __init__(self, args, base_encoder):
        super().__init__()

        self.encoder_q = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch, pretrained=True)
        # momentum encoderelf.encoder_q
        self.encoder_k = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch, pretrained=True)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(args.moco_queue, args.low_dim)) # source queue
        self.register_buffer("queue_pseudo", torch.randn(args.moco_queue, args.num_class)) # 

        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))      

        self.register_buffer("prototypes_sou", torch.zeros(args.num_class,args.low_dim)) # prototype for source domain 
        self.register_buffer("prototypes_tar", torch.zeros(args.num_class,args.low_dim)) # prototype for target domain


    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, partial_Y, args=None):
        # gather keys before updating queue
        keys = keys
        labels = labels
        partial_Y = partial_Y

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr) # 1
        assert args.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue) 维护两个队列：source queue & target queue
        # source queue
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_pseudo[ptr:ptr + batch_size, :] = labels
        ptr = (ptr + batch_size) % args.moco_queue  # move pointer annulus index
        self.queue_ptr[0] = ptr



    def reset_prototypes(self, prototypes):
        self.prototypes = prototypes

    def forward(self, img_q=None,img_k=None,partial_Y=None,img_t_q=None,img_t_k=None,p_labels_t=None,true_tar_labels=None,args=None, eval_only=False, epoch=None):
        
        with torch.no_grad():
            original_features_t, output_t, q_t = self.encoder_q(img_t_q) # target
            if eval_only:
                return original_features_t,output_t,q_t
        
        # source
        original_features_s, output_s, q_s = self.encoder_q(img_q) # source

        # update the momentum encoder
        self._momentum_update_key_encoder(args)  
    
        q_t.requires_grad = True

        original_features_sk,_,k = self.encoder_k(img_k)
 
        # source
        predicetd_scores = torch.softmax(output_s, dim=1) * partial_Y # softmax 后的预测置信度 using partial labels to filter out negative labels !!!!
        max_confi, pseudo_labels = torch.max(predicetd_scores, dim=1)
        
        # lis_ = torch.tensor([np.argmax(item) for item in partial_Y.cpu()]).cuda()
        # acc,_ = accuracy(output_s, lis_) # accuracy on source domain
        # #print('Acc source_1:', acc)
        
        # target
        #predicetd_scores_tar = F.one_hot(p_labels_t.long(), num_classes=args.num_class).float() #→ one-hot

        for feat, label in zip(q_s, pseudo_labels):
            self.prototypes_sou[label] = self.prototypes_sou[label] * args.proto_m + (1-args.proto_m) * feat
        self.prototypes_sou = F.normalize(self.prototypes_sou, p=2, dim=1)
        
        # Disambiguation source
        sou_proto_que = self.prototypes_sou.clone().detach()
        logits_prot = torch.mm(q_s, sou_proto_que.t())
        score_prot = torch.softmax(logits_prot, dim=1)


        tar_proto_que = None
        if epoch >= args.prot_start-1:
            for feat_t, label_t in zip(q_t, p_labels_t):
                self.prototypes_tar[int(label_t)] = self.prototypes_tar[int(label_t)] * args.proto_m + (1-args.proto_m) * feat_t 
            self.prototypes_tar = F.normalize(self.prototypes_tar, p=2, dim=1)
            tar_proto_que = self.prototypes_tar.clone().detach()
            # Target score_prot
            # logits_prot_tar = torch.mm()
        output_t_strong = None
        if epoch >= args.prot_start:
            original_features_tk, output_t_strong, _ = self.encoder_q(img_t_k)

        # update momentum prototypes with pseudo labels
        features = torch.cat((q_s, k, self.queue.clone().detach()), dim=0) # features is A A = q U k U queue
        feature_tar = q_t

        # pseudo-labels
        pseudo_scores = torch.cat((predicetd_scores, predicetd_scores, self.queue_pseudo.clone().detach()), dim=0)
        
        self._dequeue_and_enqueue(k, predicetd_scores, partial_Y, args=args)


        return output_s, features, feature_tar, pseudo_scores, score_prot, sou_proto_que, tar_proto_que, output_t_strong

