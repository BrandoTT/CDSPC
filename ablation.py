## Ablation Study Experiments ##
import os

## Top-1 Acc Figure curve between baselines  // Office-31
## A2W 0.09 0.16 0.22 0.29 // partial_rate

# 88.84
# -- DACO --
# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.22 --prot_start 12 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.29 --prot_start 13 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6')

## -- DACO w/o self-training --

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.22 --prot_start 12 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.29 --prot_start 13 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6')


## -- DACO w/o self-supervised --

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.0')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.0')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.22 --prot_start 12 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.0')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.29 --prot_start 13 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.0')

## self-training weight
# self-training loss weight ablation weight: 0.0, 0.2, 0.6, 0.8, 1.0

# # os.system('python -u train.py \
# #     --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
# #     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
# #     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.0')

# os.system('python -u train.py \
#     --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#     --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.4')

# os.system('python -u train.py \
#     --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.6')

# os.system('python -u train.py \
#     --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.8')

# # os.system('python -u train.py \
# #     --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
# #     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
# #     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 1.0')



## contrastive loss
# constrastive loss weight ablation weight: 0.0, 0.2, 0.6, 0.8, 1.0

# # have done
# # os.system('python -u train.py \
# #    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
# #    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
# #    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.0')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.4')

# # have done
# # os.system('python -u train.py \
# #    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
# #    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
# #    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6') 

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.8')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 1.0')

## memory bank

## Pseudo-labels Acc curve with epochs 

## T-SNE 




####### New filter strategy ########
##############
# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 10 --filter_pro_init 0.4 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_d --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_d_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 10 --filter_pro_init 0.4 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_d_a --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_w_a --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_w_d --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 10 --filter_pro_init 0.4 --weight_cont 0.6 --weight_selftrain 0.2')


# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 10 --filter_pro_init 0.4 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_a_d --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_d_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_d_a --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_w_a --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_w_d --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')


# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_d --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_d --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_d_a --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_d_a --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_d_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_d_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_w_a --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_w_a --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_w_d --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_w_d --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# # Supplenment Clean Dataset
# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.0 --prot_start 5 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_a_d --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.0 --prot_start 5 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_d_a --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.0 --prot_start 5 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_d_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.0 --prot_start 5 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_w_a --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.0 --prot_start 5 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/Ablation_Study --dataset office31_w_d --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.0 --prot_start 5 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

