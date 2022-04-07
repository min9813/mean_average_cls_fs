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

# python test.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
# python test.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512
# python test.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 256
# python test.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 128
# python test.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 64

# python test.py --dataset miniImagenet --model ResNet18 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
# python test.py --dataset miniImagenet --model ResNet18 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 128

# for model in "ResNet12" "ResNet18";
# do
# for dataset in "fc100" "cifarfs";
# do
#     for method in "baseline" "baseline++";
#     do
#         # echo $dataset $method
#         # python test.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 2048
#         # python test.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
#         # python test.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512
#         # python test.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t
#         # python test.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --test_as_protonet --normalize_method no
#         # python test.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 256
#         # python test.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 128
#         # python test.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 64

#         # python test.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 2048 
#         # python test.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
#         # python test.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512
#         python test.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t
#         python test.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --test_as_protonet --normalize_method no
#         # python test.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 128 

#     done
# done

for dataset in "miniImagenet";
do
    for method in "protonet";
    do
#         # echo $dataset $method
        python save_features.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --split base
        python save_features.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --split base
#         # python save_features.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512
#         # python save_features.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 256
#         # python save_features.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 128
#         # python save_features.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 64

#         # python save_features.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 2048 
#         # python save_features.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
#         # python save_features.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512
        python save_features.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --split base
        python save_features.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --split base

    done
done

#     for method in "no" "l2" "maha-diag_cov";
#     do
#         python test.py --dataset $dataset --model ResNet18 --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method $method
#         python test.py --dataset $dataset --model ResNet18 --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method $method
#     done

#     python test.py --dataset $dataset --model ResNet18 --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method lda_test
#     python test.py --dataset $dataset --model ResNet18 --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_test
#     python test.py --dataset $dataset --model ResNet18 --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method est_beforehand

# done

# for model in "ResNet18";
# do
#     for dataset in "miniImagenet";
#     do
#         for method in "baseline++";
#         do
        #     # echo $dataset $method
        #     # python test.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 2048
        #     # python test.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
        #     # python test.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512
        #     # python test.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t
        #     # python test.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --test_as_protonet --normalize_method no
        #     # python test.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 256
        #     # python test.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 128
        #     # python test.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 64

        #     # python test.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 2048 
        #     # python test.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
        #     # python test.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512
        #     python test.py --dataset $dataset --model $model --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512
        #     python test.py --dataset $dataset --model $model --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 512
        #     # python test.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 128 

        # done

        # for method in "force_l2-maha-diag_cov";
        # do
            # python test.py --dataset $dataset --model $model --method baseline++ --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method $method
            # python test.py --dataset $dataset --model $model --method baseline++ --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method $method
        # done

        # python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method lda_test
        # python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_test
        # python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method est_beforehand
        # python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_beforehand
        # echo "python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_beforehand_l2"
        # python test.py --dataset $dataset --model $model --method baseline++ --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_beforehand_l2

        # echo "python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method est_beforehand_l2"
        # python test.py --dataset $dataset --model $model --method baseline++ --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method est_beforehand_l2

        # echo "python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method est_beforehand"
        # python test.py --dataset $dataset --model $model --method baseline++ --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method est_beforehand

        # echo "python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method est_beforehand"
        # python test.py --dataset $dataset --model $model --method baseline++ --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method est_beforehand

        # echo "python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method lda_test"
        # python test.py --dataset $dataset --model $model --method baseline++ --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method lda_test


        # echo "python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_test"
        # python test.py --dataset $dataset --model $model --method baseline++ --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_test

            # echo "python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method force_l2-maha-diag_cov"
            # python test.py --dataset $dataset --model $model --method $method --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method force_l2-maha-diag_cov
            # echo "python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method lda_test_after_l2"
            # python test.py --dataset $dataset --model $model --method $method --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method lda_test_after_l2
            # echo "python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method lda_test_after_l2"
            # python test.py --dataset $dataset --model $model --method $method --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method maha-diag_cov
            # python test.py --dataset $dataset --model $model --method $method --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method l2
            # python test.py --dataset $dataset --model $model --method $method --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method l2
            # python test.py --dataset $dataset --model $model --method $method --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method est_beforehand_l2 --save_iter -1
            # python test.py --dataset $dataset --model $model --method $method --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_beforehand_l2 --save_iter -1
            # python test.py --dataset $dataset --model $model --method $method --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method est_beforehand --save_iter -1
            # python test.py --dataset $dataset --model $model --method $method --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_beforehand --save_iter -1

            # python test.py --dataset $dataset --model $model --method $method  --train_aug --output_dim 512 --n_shot 1
            # python test.py --dataset $dataset --model $model --method $method  --train_aug --output_dim 512 --n_shot 5
        # echo "python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_test_after_l2 --subtract_mean t --subtract_mean_method mean"
        # python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_test_after_l2 --subtract_mean t --subtract_mean_method mean

        # echo "python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_test_after_l2 --subtract_mean t --subtract_mean_method mean"
        # python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_test_after_l2 --subtract_mean t --subtract_mean_method equal_angle



        # for subtract_mean_method in "mean";
        # do
        #     echo "python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_beforehand_l2 --subtract_mean t --subtract_mean_method $subtract_mean_method"
        #     python test.py --dataset $dataset --model $model --method baseline++ --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method force_l2-est_beforehand_after_l2 --subtract_mean t --subtract_mean_method $subtract_mean_method

        #     echo "python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method est_beforehand_l2 --subtract_mean t --subtract_mean_method $subtract_mean_method"
        #     python test.py --dataset $dataset --model $model --method baseline++ --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method force_l2-est_beforehand_after_l2  --subtract_mean t --subtract_mean_method $subtract_mean_method

        #     echo "python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_test_after_l2 --subtract_mean t --subtract_mean_method $subtract_mean_method"
        #     python test.py --dataset $dataset --model $model --method baseline++ --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method force_l2-est_test_after_after_l2  --subtract_mean t --subtract_mean_method $subtract_mean_method

        #     echo "python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method l2 --subtract_mean t --subtract_mean_method $subtract_mean_method"
        #     python test.py --dataset $dataset --model $model --method baseline++ --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method l2 --subtract_mean t --subtract_mean_method $subtract_mean_method

        #     echo "python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method l2 --subtract_mean t --subtract_mean_method $subtract_mean_method"
        #     python test.py --dataset $dataset --model $model --method baseline++ --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method l2 --subtract_mean t --subtract_mean_method $subtract_mean_method
        # done


        # echo "python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method est_beforehand_l2 --subtract_mean t --subtract_mean_method mean"
        # python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method est_beforehand_l2 --subtract_mean t --subtract_mean_method mean

        # echo "python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method est_beforehand_l2 --subtract_mean t --subtract_mean_method equal_angle"
        # python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method est_beforehand_l2 --subtract_mean t --subtract_mean_method equal_angle
        # python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_beforehand_l2_est_test
        # python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_beforehand_l2
        # python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_beforehand_l2_est_test
        # python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_beforehand_l2
        # python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method est_beforehand_l2_est_test
        # python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method est_beforehand_l2
        # python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_beforehand_l2_est_test
        # python test.py --dataset $dataset --model $model --method baseline --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method est_beforehand_l2
    # done
# done

for model in "ResNet12" "ResNet18";
do
    for dataset in "miniImagenet";
    do
        # for method in "baseline++";
        for method in "protonet";
        do
        # python test.py --dataset $dataset --model $model --method $method --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method l2
        # python test.py --dataset $dataset --model $model --method $method --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method l2
        # python test.py --dataset $dataset --model $model --method $method --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method no
        # python test.py --dataset $dataset --model $model --method $method --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method no
        python test.py --dataset $dataset --model $model --method $method --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method force_l2-est_beforehand_after_l2
        python test.py --dataset $dataset --model $model --method $method --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method force_l2-est_beforehand_after_l2
        # python test.py --dataset $dataset --model $model --method $method --test_as_protonet --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --normalize_method force_l2-est_beforehand_after_l2
        # python test.py --dataset $dataset --model $model --method $method --test_as_protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --normalize_method force_l2-est_beforehand_after_l2

        done
    done
done
