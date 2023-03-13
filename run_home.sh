#####################
# === Office-Home === 
#####################

# # Ar2Cl 
# pr           - 0.0 - 0.05 - 0.07
# protostart   -  5  -  11  - 
# learning rate: 0.001 decay lr epoch 200

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.0 --prot_start 5 --print-freq 5 --filter_pro_init 0.2 --weight_cont 0.6

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
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 15 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# # Ar2Pr
# pr           - 0.0 - 0.05 - 0.07
# protostart   -  15  -    - 
# learning rate: 0.001 decay lr epoch 200
# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 11 --print-freq 10 --filter_pro_init 0.1 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 15 --print-freq 10 --filter_pro_init 0.1 --weight_cont 0.6


# # Ar2Rw
# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Ar_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 15 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# # Cl2Pr

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 15 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6


# # Cl2Rw

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 15 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# # # Cl2Ar
# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Cl_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 15 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# # # Pr2Ar

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 15 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# # # Pr2Rw
# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Rw --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 15 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# # # Pr2Cl
# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Pr_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 15 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# # # Rw2Ar
# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Ar --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 15 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# # # Rw2Cl
# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Cl --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 15 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# # # Rw2Pr
# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.05 --prot_start 11 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

# python -u train.py \
#    --exp-dir experiment/office_home --dataset officehome_Rw_Pr --num-class 65 --batch-size 32\
#    --seed 123 --arch resnet50 --moco_queue 2048 --lr 0.001 --wd 1e-3 --cosine --epochs 100 \
#    --loss_weight 0.2 --proto_m 0.99 --partial_rate 0.07 --prot_start 15 --print-freq 10 --filter_pro_init 0.2 --weight_cont 0.6

