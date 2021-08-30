# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug
# python test.py --dataset miniImagenet --model ResNet18 --method baseline++
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++

# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --support_aug --support_aug_num 5 # 90.16 +- 0.38
# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --support_aug --support_aug_num 3
# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --support_aug --support_aug_num 10
# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --support_aug --support_aug_num 10
# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --support_aug --support_aug_num 5
# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --support_aug --support_aug_num 3
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --support_aug --support_aug_num 10
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --support_aug --support_aug_num 5
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --support_aug --support_aug_num 3
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --support_aug --support_aug_num 10
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --support_aug --support_aug_num 5
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --support_aug --support_aug_num 3

# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --test_as_protonet --normalize_method l2 # 90.16 +- 0.38
# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --test_as_protonet --normalize_method l2 # 90.16 +- 0.38
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --test_as_protonet --normalize_method l2 # 90.16 +- 0.38
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --test_as_protonet --normalize_method l2 # 90.16 +- 0.38
# python test.py --dataset miniImagenet --model ResNet18 --method baseline --train_aug --test_as_protonet --normalize_method l2 # 90.16 +- 0.38
# python test.py --dataset miniImagenet --model ResNet18 --method baseline --test_as_protonet --normalize_method l2 # 90.16 +- 0.38

# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --support_aug --support_aug_num 3
# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --support_aug --support_aug_num 10
# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --support_aug --support_aug_num 10
# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --support_aug --support_aug_num 5
# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --support_aug --support_aug_num 3
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --support_aug --support_aug_num 10
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --support_aug --support_aug_num 5
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --support_aug --support_aug_num 3
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --support_aug --support_aug_num 10
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --support_aug --support_aug_num 5
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --support_aug --support_aug_num 3

# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 128 --test_as_protonet --normalize_method l2
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 128 --test_as_protonet --normalize_method no
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 256 --test_as_protonet --normalize_method l2
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 256 --test_as_protonet --normalize_method no
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 1024 --test_as_protonet --normalize_method l2
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 1024 --test_as_protonet --normalize_method no

# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --output_dim 128 --test_as_protonet --normalize_method l2
# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --output_dim 128 --test_as_protonet --normalize_method no
# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --output_dim 256 --test_as_protonet --normalize_method l2
# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --output_dim 256 --test_as_protonet --normalize_method no
# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --output_dim 1024 --test_as_protonet --normalize_method l2
# python test.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --output_dim 1024 --test_as_protonet --normalize_method no
# python save_features.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --output_dim 128
# python save_features.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 256
# python save_features.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --output_dim 256
# python save_features.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 1024
# python save_features.py --dataset miniImagenet --model ResNet18 --method baseline++ --train_aug --output_dim 1024
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector before --normalize_query t
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector before_and_mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector before --normalize_query f
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector before_and_mean --normalize_query f
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query f

# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector before --normalize_query t --norm_factor 4
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector before_and_mean --normalize_query t --norm_factor 4
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t --norm_factor 4
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector before --normalize_query f --norm_factor 4
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector before_and_mean --normalize_query f --norm_factor 4
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query f --norm_factor 4

# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 256 --test_as_protonet --normalize_method l2 --normalize_vector before --normalize_query t
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 256 --test_as_protonet --normalize_method l2 --normalize_vector before_and_mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 256 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 256 --test_as_protonet --normalize_method l2 --normalize_vector before --normalize_query f
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 256 --test_as_protonet --normalize_method l2 --normalize_vector before_and_mean --normalize_query f
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 256 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query f

# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 256 --test_as_protonet --normalize_method l2 --normalize_vector before --normalize_query t --norm_factor 4
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 256 --test_as_protonet --normalize_method l2 --normalize_vector before_and_mean --normalize_query t --norm_factor 4
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 256 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t --norm_factor 4
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 256 --test_as_protonet --normalize_method l2 --normalize_vector before --normalize_query f --norm_factor 4
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 256 --test_as_protonet --normalize_method l2 --normalize_vector before_and_mean --normalize_query f --norm_factor 4
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 256 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query f --norm_factor 4

# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 128 --test_as_protonet --normalize_method l2 --normalize_vector before --normalize_query t
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 128 --test_as_protonet --normalize_method l2 --normalize_vector before_and_mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 128 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 128 --test_as_protonet --normalize_method l2 --normalize_vector before --normalize_query f
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 128 --test_as_protonet --normalize_method l2 --normalize_vector before_and_mean --normalize_query f
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 128 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query f

# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 128 --test_as_protonet --normalize_method l2 --normalize_vector before --normalize_query t --norm_factor 4
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 128 --test_as_protonet --normalize_method l2 --normalize_vector before_and_mean --normalize_query t --norm_factor 4
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 128 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t --norm_factor 4
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 128 --test_as_protonet --normalize_method l2 --normalize_vector before --normalize_query f --norm_factor 4
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 128 --test_as_protonet --normalize_method l2 --normalize_vector before_and_mean --normalize_query f --norm_factor 4
# python test.py --dataset miniImagenet --model ResNet12 --method baseline++ --train_aug --output_dim 128 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query f --norm_factor 4

# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 1024 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 1024 --test_as_protonet --normalize_method no --normalize_vector mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 512 --test_as_protonet --normalize_method no --normalize_vector mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 256 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 256 --test_as_protonet --normalize_method no --normalize_vector mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 128 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 128 --test_as_protonet --normalize_method no --normalize_vector mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 64 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 64 --test_as_protonet --normalize_method no --normalize_vector mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet18 --method baseline --train_aug --output_dim 1024 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet18 --method baseline --train_aug --output_dim 1024 --test_as_protonet --normalize_method no --normalize_vector mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet18 --method baseline --train_aug --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t
# python test.py --dataset miniImagenet --model ResNet18 --method baseline --train_aug --output_dim 512 --test_as_protonet --normalize_method no --normalize_vector mean --normalize_query t

# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 64 --normalize_vector mean --n_shot 1 --loss_type dist
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 128 --normalize_vector mean --n_shot 1 --loss_type dist
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 256 --normalize_vector mean --n_shot 1 --loss_type dist
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 512 --normalize_vector mean --n_shot 1 --loss_type dist
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 1024 --normalize_vector mean --n_shot 1 --loss_type dist
# python test.py --dataset miniImagenet --model ResNet18 --method baseline --train_aug --output_dim 512 --normalize_vector mean --n_shot 1 --loss_type dist
# python test.py --dataset miniImagenet --model ResNet18 --method baseline --train_aug --output_dim 1024 --normalize_vector mean --n_shot 1 --loss_type dist

# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 64 --normalize_vector mean --n_shot 10 --loss_type dist
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 128 --normalize_vector mean --n_shot 10 --loss_type dist
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 256 --normalize_vector mean --n_shot 10 --loss_type dist
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 512 --normalize_vector mean --n_shot 10 --loss_type dist
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 1024 --normalize_vector mean --n_shot 10 --loss_type dist
# python test.py --dataset miniImagenet --model ResNet18 --method baseline --train_aug --output_dim 512 --normalize_vector mean --n_shot 10 --loss_type dist
# python test.py --dataset miniImagenet --model ResNet18 --method baseline --train_aug --output_dim 1024 --normalize_vector mean --n_shot 10 --loss_type dist

# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 1024 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t --n_shot 10
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 1024 --test_as_protonet --normalize_method no --normalize_vector mean --normalize_query t --n_shot 10
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t --n_shot 10
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 512 --test_as_protonet --normalize_method no --normalize_vector mean --normalize_query t --n_shot 10
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 256 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t --n_shot 10
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 256 --test_as_protonet --normalize_method no --normalize_vector mean --normalize_query t --n_shot 10
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 128 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t --n_shot 10
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 128 --test_as_protonet --normalize_method no --normalize_vector mean --normalize_query t --n_shot 10
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 64 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t --n_shot 10
# python test.py --dataset miniImagenet --model ResNet12 --method baseline --train_aug --output_dim 64 --test_as_protonet --normalize_method no --normalize_vector mean --normalize_query t --n_shot 10
# python test.py --dataset miniImagenet --model ResNet18 --method baseline --train_aug --output_dim 1024 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t --n_shot 10
# python test.py --dataset miniImagenet --model ResNet18 --method baseline --train_aug --output_dim 1024 --test_as_protonet --normalize_method no --normalize_vector mean --normalize_query t --n_shot 10
# python test.py --dataset miniImagenet --model ResNet18 --method baseline --train_aug --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t --n_shot 10
# python test.py --dataset miniImagenet --model ResNet18 --method baseline --train_aug --output_dim 512 --test_as_protonet --normalize_method no --normalize_vector mean --normalize_query t --n_shot 10

python test.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
python test.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512
python test.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 256
python test.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 128
python test.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 64

python test.py --dataset miniImagenet --model ResNet18 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
python test.py --dataset miniImagenet --model ResNet18 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 128