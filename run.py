# Demo shell scripts
# Note that we only tested the single-GPU version
# Please carefully check the code if you would like to use multiple GPUs
import os
# === mnist → syndigits p = 0.0 
# 2022-05-25 best acc 93.4%
# python -u train.py \
#    --exp-dir experiment/mnist_syndigits --dataset m2s --num-class 10\
#    --seed 123 --arch resnet18 --moco_queue 8192 --lr 0.01 --wd 1e-3 --cosine --epochs 60 \
#    --loss_weight 0.5 --proto_m 0.99 --partial_rate 0.0 --prot_start 30 --print-freq 20 --filter_pro_init 0.2 

# p=0.0  93.40%
# p=0.3  93.45%
# p=0.5  89.43%
# python -u train.py \
#    --exp-dir experiment/mnist_syndigits --dataset m2s --num-class 10 --batch-size 64\
#    --seed 3407 --arch resnet18 --moco_queue 8192 --lr 0.01 --wd 1e-3 --cosine --epochs 40 \
#    --loss_weight 0.5 --proto_m 0.99 --partial_rate 0.3 --prot_start 14 --print-freq 20 --filter_pro_init 0.2 --weight_cont 0.6

# === syndigits → mnist 
# p=0.0 95.41%
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
#   --exp-dir experiment/mnist_syndigits --dataset s2m --num-class 10\
#   --seed 123 --arch resnet18 --moco_queue 8192 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#   --loss_weight 0.5 --proto_m 0.99 --partial_rate 0.0 --prot_start 14 --print-freq 20 --filter_pro_init 0.2

# p=0.3  95.02%
# python -u train.py \
#    --exp-dir experiment/mnist_syndigits --dataset s2m --num-class 10\
#    --seed 123 --arch resnet18 --moco_queue 8192 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.5 --proto_m 0.99 --partial_rate 0.3 --prot_start 14 --print-freq 20 --filter_pro_init 0.2 

#p=0.5 92.70 %
# python -u train.py \
#    --exp-dir experiment/mnist_syndigits --dataset s2m --num-class 10\
#    --seed 123 --arch resnet18 --moco_queue 8192 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.5 --proto_m 0.99 --partial_rate 0.5 --prot_start 14 --print-freq 20 --filter_pro_init 0.2 

####################
# === Office-31 === 
####################
# A → W
# best: 86.31
# python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.0 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6


# 2022 9 24 adjusted best result batch = 32
## best: 87.32
# python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 1 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

## best: 87.95
# python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_a_w --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 4 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6



# A → D
# best: 
# python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_a_w --num-class 31 --batch-size 128\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.0 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# best: 89.95
# os.system('python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_a_d --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# # best: 74.82
# python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_a_d --num-class 31 --batch-size 64\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# # W → A
# # best: 
# python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_w_a --num-class 31 --batch-size 128\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# # best: 
# python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_w_a --num-class 31 --batch-size 128\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6


# W → D
# # best: 100.0
# python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_w_d --num-class 31 --batch-size 64\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# # best: 99.60
# python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_w_d --num-class 31 --batch-size 64\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6


# # D → W
# # best: 
# python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_d_w --num-class 31 --batch-size 64\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# # best: 
# python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_d_w --num-class 31 --batch-size 64\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# # D → A 
# best: 
# python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_d_a --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 200 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.7

# os.system('python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_d_a --num-class 31 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.09 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# # best: 
# python -u train.py \
#    --exp-dir experiment/office_31 --dataset office31_d_a --num-class 31 --batch-size 128\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6


####################
# === Office-Home === 
####################


# # Ar2Cl 
# pr           - 0.0 - 0.05 - 0.07
# protostart   -  5  -  11  - 
# learning rate: 0.001 decay lr epoch 200

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 11 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6')

#prostart: 1 -> 3 (58.26) -> deleate cosine (57.71) -> protostart + (cosine)5 61.89 
# learning rate decay epoch: 200

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# learning rate decay epoch: 200/100 -> 60/100 (51.7)
# protostart 9 55.31 -> protostart 11 55.98

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# # Ar2Pr
# pr           - 0.0 - 0.05 - 0.07
# protostart   -  15  -    - 
# learning rate: 0.001 decay lr epoch 200
# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.0 --prot_start 15 --print-freq 10 --filter_pro_init 0.1 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# # Ar2Rw

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.0 --prot_start 5 --print-freq 15 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# # Cl2Pr

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.04 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6


# # Cl2Rw

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.04 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# # Cl2Ar
# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.04 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# # Pr2Ar

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.04 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# # Pr2Rw
# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.04 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# # Pr2Cl
# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.04 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# # Rw2Ar
# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.04 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# # Rw2Cl
# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.04 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# # Rw2Pr
# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.04 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.01 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 10 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

# # supplement without Dis

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')


# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')


# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')



####################
# === VisDa 2017 === 
####################
# import os

# os.system('python -u train.py \
#    --exp-dir experiment/visda17 --dataset visda2017 --num-class 12 --batch-size 64\
#    --seed 123 --arch resnet101 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.25 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6')

# python -u train.py \
#    --exp-dir experiment/visda17 --dataset visda2017 --num-class 12 --batch-size 16\
#    --seed 123 --arch resnet101 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.5 --prot_start 15 --print-freq 100 --filter_pro_init 0.2 --weight_cont 0.6



############### ---------- Office New Filter Strategy----------------- ###################

# Office-31 pr: 0.15, 0.2, 0.3

# os.system('python -u train.py \
#     --exp-dir experiment/office_31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.2 --prot_start 12 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#     --exp-dir experiment/office_31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.3 --prot_start 13 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#     --exp-dir experiment/office_31 --dataset office31_a_d --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.2 --prot_start 12 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#     --exp-dir experiment/office_31 --dataset office31_a_d --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.3 --prot_start 13 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#     --exp-dir experiment/office_31 --dataset office31_d_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.2 --prot_start 12 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#     --exp-dir experiment/office_31 --dataset office31_d_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.3 --prot_start 13 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#     --exp-dir experiment/office_31 --dataset office31_d_a --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.2 --prot_start 12 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#     --exp-dir experiment/office_31 --dataset office31_d_a --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.3 --prot_start 13 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#     --exp-dir experiment/office_31 --dataset office31_w_a --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.2 --prot_start 12 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#     --exp-dir experiment/office_31 --dataset office31_w_a --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.3 --prot_start 13 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')


# os.system('python -u train.py \
#     --exp-dir experiment/office_31 --dataset office31_w_d --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.2 --prot_start 12 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#     --exp-dir experiment/office_31 --dataset office31_w_d --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.3 --prot_start 13 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')


# Office-Home pr: 0.1, 0.15

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.1 --prot_start 13 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# # os.system('python -u train.py \
# #    --exp-dir experiment/office_home --dataset officehome_Ar_Cl --num-class 65 --batch-size 32\
# #    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
# #    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.15 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.1 --prot_start 13 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.1 --prot_start 13 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')


# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.1 --prot_start 13 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')


# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.1 --prot_start 13 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')


# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.1 --prot_start 13 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# # os.system('python -u train.py \
# #    --exp-dir experiment/office_home --dataset officehome_Cl_Ar --num-class 65 --batch-size 32\
# #    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
# #    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.15 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.1 --prot_start 13 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.1 --prot_start 13 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.1 --prot_start 13 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# # os.system('python -u train.py \
# #    --exp-dir experiment/office_home --dataset officehome_Pr_Ar --num-class 65 --batch-size 32\
# #    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
# #    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.15 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.1 --prot_start 13 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.1 --prot_start 13 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.1 --prot_start 13 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# # os.system('python -u train.py \
# #    --exp-dir experiment/office_home --dataset officehome_Rw_Ar --num-class 65 --batch-size 32\
# #    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
# #    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.15 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')







# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# # os.system('python -u train.py \
# #    --exp-dir experiment/office_home --dataset officehome_Ar_Cl --num-class 65 --batch-size 32\
# #    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
# #    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.055 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')


# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')


# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')


# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# # os.system('python -u train.py \
# #    --exp-dir experiment/office_home --dataset officehome_Cl_Ar --num-class 65 --batch-size 32\
# #    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
# #    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.055 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# # os.system('python -u train.py \
# #    --exp-dir experiment/office_home --dataset officehome_Pr_Ar --num-class 65 --batch-size 32\
# #    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
# #    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.055 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 10 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# # os.system('python -u train.py \
# #    --exp-dir experiment/office_home --dataset officehome_Rw_Ar --num-class 65 --batch-size 32\
# #    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
# #    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.15 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')


#  0.15



# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.15 --prot_start 14 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# # os.system('python -u train.py \
# #    --exp-dir experiment/office_home --dataset officehome_Ar_Cl --num-class 65 --batch-size 32\
# #    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
# #    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.155 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.15 --prot_start 14 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.15 --prot_start 14 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')


# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.15 --prot_start 14 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')


# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.15 --prot_start 14 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')


# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.15 --prot_start 14 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# # os.system('python -u train.py \
# #    --exp-dir experiment/office_home --dataset officehome_Cl_Ar --num-class 65 --batch-size 32\
# #    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
# #    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.155 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.15 --prot_start 14 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.15 --prot_start 14 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.15 --prot_start 14 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# # os.system('python -u train.py \
# #    --exp-dir experiment/office_home --dataset officehome_Pr_Ar --num-class 65 --batch-size 32\
# #    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
# #    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.155 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.15 --prot_start 14 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.15 --prot_start 14 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# os.system('python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.15 --prot_start 14 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')
# # os.system('python -u train.py \
# #    --exp-dir experiment/office_home --dataset officehome_Rw_Ar --num-class 65 --batch-size 32\
# #    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
# #    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.15 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')


# supplyment ablation experiments
# self-training loss weight
# Office-31 A → W 0.16
# os.system('python -u train.py \
#     --exp-dir experiment/New/office31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.1 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#     --exp-dir experiment/New/office31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#     --exp-dir experiment/New/office31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.3 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#     --exp-dir experiment/New/office31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.4 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#     --exp-dir experiment/New/office31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.5 --weight_cont 0.6 --weight_selftrain 0.2')

### self-training weight (selftrainW)
# os.system('python -u train.py \
#     --exp-dir experiment/New/office31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.0')

# os.system('python -u train.py \
#     --exp-dir experiment/New/office31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#     --exp-dir experiment/New/office31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.4')

# os.system('python -u train.py \
#     --exp-dir experiment/New/office31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.6')

# os.system('python -u train.py \
#     --exp-dir experiment/New/office31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.8')

# os.system('python -u train.py \
#     --exp-dir experiment/New/office31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 1.0')


### self-supervised weight (selfsupervisedW)
# os.system('python -u train.py \
#     --exp-dir experiment/New/office31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.0 --weight_selftrain 0.2')

# os.system('python -u train.py \
#     --exp-dir experiment/New/office31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.2 --weight_selftrain 0.2')

# os.system('python -u train.py \
#     --exp-dir experiment/New/office31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.4 --weight_selftrain 0.2')

# os.system('python -u train.py \
#     --exp-dir experiment/New/office31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#     --exp-dir experiment/New/office31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.8 --weight_selftrain 0.2')

# os.system('python -u train.py \
#     --exp-dir experiment/New/office31 --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 1.0 --weight_selftrain 1.0')



### Class-Wise Accuracy office31 A→W pr0.16， 0.2
# os.system('python -u train.py \
#     --exp-dir experiment/New/feature_ablation --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#     --exp-dir experiment/New/feature_ablation --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.2 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# T-SNE 
# for study consistent loss & contrastive loss

# w/o consistent Loss
# os.system('python -u train.py \
#     --exp-dir experiment/New/feature_ablation --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.0')

# os.system('python -u train.py \
#     --exp-dir experiment/New/feature_ablation --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.2 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.0')

# w/o contrastive Loss
# os.system('python -u train.py \
#     --exp-dir experiment/New/feature_ablation --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.0 --weight_selftrain 0.2')

# os.system('python -u train.py \
#     --exp-dir experiment/New/feature_ablation --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.2 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.0 --weight_selftrain 0.2')

### ablation for pseudo-labels
# os.system('python -u train.py \
#     --exp-dir experiment/New/feature_ablation --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# os.system('python -u train.py \
#     --exp-dir experiment/New/feature_ablation --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.2 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2')

# Final Ablation Study Office-31 partial 0.16
# just loss_class source
# os.system('python -u train.py \
#     --exp-dir experiment/New/final_ablation --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.0 --weight_cont 0.0 --weight_selftrain 0.0')
# os.system('python -u train.py \
#     --exp-dir experiment/New/final_ablation --dataset office31_a_d --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.0 --weight_cont 0.0 --weight_selftrain 0.0')
# os.system('python -u train.py \
#     --exp-dir experiment/New/final_ablation --dataset office31_d_a --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.0 --weight_cont 0.0 --weight_selftrain 0.0')
# os.system('python -u train.py \
#     --exp-dir experiment/New/final_ablation --dataset office31_d_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.0 --weight_cont 0.0 --weight_selftrain 0.0')
# os.system('python -u train.py \
#     --exp-dir experiment/New/final_ablation --dataset office31_w_a --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.0 --weight_cont 0.0 --weight_selftrain 0.0')
# os.system('python -u train.py \
#     --exp-dir experiment/New/final_ablation --dataset office31_w_d --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.0 --weight_cont 0.0 --weight_selftrain 0.0')
# add loss_cont & loss_t
# os.system('python -u train.py \
#     --exp-dir experiment/New/final_ablation --dataset office31_a_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2 --weight_neg 0.0')
# os.system('python -u train.py \
#     --exp-dir experiment/New/final_ablation --dataset office31_a_d --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2 --weight_neg 0.0')
# os.system('python -u train.py \
#     --exp-dir experiment/New/final_ablation --dataset office31_d_a --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2 --weight_neg 0.0')
# os.system('python -u train.py \
#     --exp-dir experiment/New/final_ablation --dataset office31_d_w --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2 --weight_neg 0.0')
# os.system('python -u train.py \
#     --exp-dir experiment/New/final_ablation --dataset office31_w_a --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2 --weight_neg 0.0')
# os.system('python -u train.py \
#     --exp-dir experiment/New/final_ablation --dataset office31_w_d --num-class 31 --batch-size 32\
#     --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#     --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2 --weight_neg 0.0')
# L_Neg
os.system('python -u train.py \
    --exp-dir experiment/New/final_ablation --dataset office31_a_w --num-class 31 --batch-size 32\
    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2 --weight_neg 1.0')
os.system('python -u train.py \
    --exp-dir experiment/New/final_ablation --dataset office31_a_d --num-class 31 --batch-size 32\
    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2 --weight_neg 1.0')
os.system('python -u train.py \
    --exp-dir experiment/New/final_ablation --dataset office31_d_a --num-class 31 --batch-size 32\
    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2 --weight_neg 1.0')
os.system('python -u train.py \
    --exp-dir experiment/New/final_ablation --dataset office31_d_w --num-class 31 --batch-size 32\
    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2 --weight_neg 1.0')
os.system('python -u train.py \
    --exp-dir experiment/New/final_ablation --dataset office31_w_a --num-class 31 --batch-size 32\
    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2 --weight_neg 1.0')
os.system('python -u train.py \
    --exp-dir experiment/New/final_ablation --dataset office31_w_d --num-class 31 --batch-size 32\
    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.16 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6 --weight_selftrain 0.2 --weight_neg 0.0')
# add 基于原型纠正的伪标记策略

