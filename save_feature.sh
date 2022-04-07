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

# python save_features.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
# python save_features.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 512
# python save_features.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 256
# python save_features.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 128
# python save_features.py --dataset miniImagenet --model ResNet12 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 64

# python save_features.py --dataset miniImagenet --model ResNet18 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
# python save_features.py --dataset miniImagenet --model ResNet18 --method protonet --train_aug --train_n_way 20 --n_shot 5 --output_dim 128

# model="ResNet18"
# python save_features.py --dataset miniImagenet --model $model --split base --method baseline --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --test_as_protonet --normalize_method l2 --normalize_vector mean --normalize_query t --add_final_layer f

# for dataset in "fc100" "cifarfs";
for dataset in "cifarfs";
do
    for method in "protonet";
    do
        # echo $dataset $method
        # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 2048
        # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 1024
        python save_features.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --add_final_layer f
        python save_features.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --add_final_layer f
        # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --add_final_layer f
        # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 256
        # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 128
        # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 64
        # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --add_final_layer f
        # python train.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --add_final_layer f

        # python train.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 2048 
        # python train.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 1024
        # python train.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --add_final_layer t
        # python train.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 128 

    done

    for method in "baseline" "baseline++";
    do
        python save_features.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --add_final_layer f
        python save_features.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --add_final_layer f
    done
done

# for dataset in "tieredImagenet";
# do
#     for method in "protonet" "baseline" "baseline++";
#     do
#         # echo $dataset $method
#         # python save_features.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 2048
#         # python save_features.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
#         python save_features.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512
#         # python save_features.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 256
#         # python save_features.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 128
#         # python save_features.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 64

#         # python save_features.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 2048 
#         # python save_features.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
#         python save_features.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512
#         # python save_features.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 128 

#         if [ $method = "protonet" ];
#         then
#             python save_features.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 512
#             python save_features.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 512

#         fi

#     done
# done