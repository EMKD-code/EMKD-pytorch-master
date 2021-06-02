# train student model with SSKD only
CUDA_VISIBLE_DEVICES=0 python student.py --t-path ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --t-arch resnet110 --s-arch resnet20 --distill sskd --lr 0.05 --gpu-id 0 --trial 1

# train student model with SSKD and MC
CUDA_VISIBLE_DEVICES=0 python student.py --t-path ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --t-arch resnet110 --s-arch myresnet20 --distill sskd --mc 1 --depth 3,3,3 --lr 0.05 --gpu-id 0 --trial 1
