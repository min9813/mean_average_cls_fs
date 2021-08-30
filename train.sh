# python train.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug
# python train.py --dataset miniImagenet --model ResNet12 --method baseline++
# python train.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug
# python train.py --dataset miniImagenet --model ResNet18 --method baseline++
# python train.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 128
# python train.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --output_dim 128
# python train.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 256
# python train.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --output_dim 256
# python train.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 1024
# python train.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --output_dim 1024
# python train.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 2048
# python train.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --output_dim 2048

# python train.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 1024
# python train.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 256
# python train.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 128
# python train.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 64
# python train.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 64
# python train.py --dataset miniImagenet --model ResNet18 --method baseline --train_aug --output_dim 1024
# python train.py --dataset miniImagenet --model ResNet18 --method baseline --train_aug --output_dim 128

python train.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
python train.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512
python train.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 256
python train.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 128
python train.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 64

python train.py --dataset miniImagenet --model ResNet18 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
python train.py --dataset miniImagenet --model ResNet18 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 128
