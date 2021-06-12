# train student model using distillation

# train student model with KD only
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -a 0 -b 0.8 --trial 1
# train student model with KD and MC
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --mc 1 --mc_weight 0.001 --model_s myresnet20 --block_depth 3,7,2 -a 0 -b 0.8 --trial 1

# train student model with Fitnet only
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1
# train student model with Fitnet and MC
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --mc 1 --mc_weight 0.001 --model_s myresnet20 --block_depth 3,7,2 -a 0 -b 100 --trial 1

# train student model with AT only
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill attention --model_s resnet20 -a 0 -b 1000 --trial 1
# train student model with AT and MC
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill attention --mc 1 --mc_weight 0.0001 --model_s myresnet20 --block_depth 3,7,2 -a 0 -b 1000 --trial 1

# train student model with SP only
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill similarity --model_s resnet20 -a 0 -b 3000 --trial 1
# train student model with SP and MC
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill similarity --mc 1 --mc_weight 0.00001 --model_s myresnet20 --block_depth 3,7,2 -a 0 -b 3000 --trial 1

# train student model with CC only
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill correlation --model_s resnet20 -a 0 -b 0.02 --trial 1
# train student model with CC and MC
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill correlation --mc 1 --mc_weight 0.001 --model_s myresnet20 --block_depth 3,7,2 -a 0 -b 0.02 --trial 1

# train student model with VID only
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill vid --model_s resnet20 -a 0 -b 1 --trial 1
# train student model with VID and MC
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill vid --mc 1 --mc_weight 0.001 --model_s myresnet20 --block_depth 3,7,2 -a 0 -b 1 --trial 1

# train student model with RKD only
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill rkd --model_s resnet20 -a 0 -b 1 --trial 1
# train student model with RKD and MC
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill rkd --mc 1 --mc_weight 0.001 --model_s myresnet20 --block_depth 3,7,2 -a 0 -b 1 --trial 1

# train student model with PKT only
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet20 -a 0 -b 30000 --trial 1
# train student model with PKT and MC
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill pkt --mc 1 --mc_weight 0.000001 --model_s myresnet20 --block_depth 3,7,2 -a 0 -b 30000 --trial 1

# train student model with AB only
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill abound --model_s resnet20 -a 0 -b 1 --trial 1
# train student model with AB and MC
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill abound --mc 1 --mc_weight 0.001 --model_s myresnet20 --block_depth 3,7,2 -a 0 -b 1 --trial 1

# train student model with FT only
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill factor --model_s resnet20 -a 0 -b 200 --trial 1
# train student model with FT and MC
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill factor --mc 1 --mc_weight 0.001 --model_s myresnet20 --block_depth 3,7,2 -a 0 -b 200 --trial 1

# train student model with FSP only
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill fsp --model_s resnet20 -a 0 -b 50 --trial 1
# train student model with FSP and MC
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill fsp --mc 1 --mc_weight 0.001 --model_s myresnet20 --block_depth 3,7,2 -a 0 -b 50 --trial 1

# train student model with NST only
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill nst --model_s resnet20 -a 0 -b 50 --trial 1
# train student model with NST and MC
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill nst --mc 1 --mc_weight 0.001 --model_s myresnet20 --block_depth 3,7,2 -a 0 -b 50 --trial 1

# train student model with CRD only
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet20 -a 0 -b 0.8 --trial 1
# train student model with CRD and MC
CUDA_VISIBLE_DEVICES=0 python train_student.py --model_t resnet110 --path_t ./checkpoints/cifar100/resnet110_vanilla/ckpt_epoch_240.pth --distill crd --mc 1 --mc_weight 0.001 --model_s myresnet20 --block_depth 3,7,2 -a 0 -b 0.8 --trial 1
