# python save_features.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug
# python save_features.py --dataset miniImagenet --model ResNet18 --method baseline++
# python save_features.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug
# python save_features.py --dataset miniImagenet --model ResNet12 --method baseline++

# python save_features.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 128
# python save_features.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --output_dim 128
# python save_features.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 256
# python save_features.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --output_dim 256
# python save_features.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 1024
# python save_features.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --output_dim 1024


# python save_features.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 1024
# python save_features.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 512 --split base
# python save_features.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 256
# python save_features.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 128
# python save_features.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 64
# python save_features.py --dataset miniImagenet --model ResNet18 --method baseline --train_aug --output_dim 1024
# python save_features.py --dataset miniImagenet --model ResNet18 --method baseline --train_aug --output_dim 512

# python save_features.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --output_dim 512 --split base
# python save_features.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
# python save_features.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512

python save_features.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
python save_features.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512
python save_features.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 256
python save_features.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 128
python save_features.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 64

python save_features.py --dataset miniImagenet --model ResNet18 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
python save_features.py --dataset miniImagenet --model ResNet18 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 128