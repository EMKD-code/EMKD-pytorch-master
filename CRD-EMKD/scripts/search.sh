# search for a student with best architecture with BCS

# search for best student architecture using CRD only
CUDA_VISIBLE_DEVICES=0 python search.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill crd --model_s myresnet20 --block_depth 3,3,3 -a 0 -b 0.8 --trial 1

# search for best student architecture using CRD and MC
CUDA_VISIBLE_DEVICES=0 python search.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill crd --mc 1 --model_s myresnet20 --block_depth 3,3,3 -a 0 -b 0.8 --trial 1
