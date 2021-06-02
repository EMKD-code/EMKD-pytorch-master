# search for a student with best architecture with BCS

# search for best student architecture with SSKD only
CUDA_VISIBLE_DEVICES=0 python search.py --t-path ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --t-arch resnet110 --s-arch myresnet20 --distill sskd --depth 3,3,3 --lr 0.05 --gpu-id 0 --trial 1

# search for best student architecture with SSKD and MC
CUDA_VISIBLE_DEVICES=0 python search.py --t-path ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --t-arch resnet110 --s-arch myresnet20 --distill sskd --mc 1 --depth 3,3,3 --lr 0.05 --gpu-id 0 --trial 1
