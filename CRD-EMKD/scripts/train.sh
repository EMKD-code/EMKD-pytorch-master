# train student model using distillation

# train student model with CRD only
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet20 -a 0 -b 0.8 --trial 1

# train student model with CRD and MC
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill crd --mc 1 --mc_weight 0.001 --model_s myresnet20 --block_depth 3,7,2 -a 0 -b 0.8 --trial 1
