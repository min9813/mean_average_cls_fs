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
for dataset in "cifarfs";
do
    for method in "protonet";
    do
        # echo $dataset $method
        # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 2048
        # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 1024
        python train.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --add_final_layer f --n_episode 400
        python train.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --add_final_layer f --n_episode 400
        # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --add_final_layer f --n_episode 400
        # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 256
        # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 128
        # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 64
        # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --add_final_layer f --n_episode 400
        # python train.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --add_final_layer f --n_episode 400

        # python train.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 2048 
        # python train.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 1024
        # python train.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --add_final_layer t
        # python train.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 128 

    done

    for method in "baseline" "baseline++";
    do
        python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --add_final_layer f --n_episode 400
        python train.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --add_final_layer f --n_episode 400
    done
done

# for dataset in "tieredImagenet";
# do
#     # for method in "protonet";
#     for method in "baseline" "baseline++" "protonet";
#     do
#         # echo $dataset $method
#         # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 2048
#         # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
#         python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --n_episode 500 --add_final_layer t
#         # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 256
#         # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 128
#         # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 64

#         # python train.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 2048 
#         # python train.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
#         python train.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 512 --n_episode 500 --add_final_layer t
#         # python train.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --n_episode 500
#         # python train.py --dataset $dataset --model ResNet12 --method $method --train_aug --train_n_way 20 --n_shot 1 --output_dim 512 --n_episode 500
#         # python train.py --dataset $dataset --model ResNet18 --method $method --train_aug --train_n_way 20 --n_shot 5 --output_dim 128 

#     done
# done
    # python train.py --dataset miniImagenet --model ResNet34 --method baseline --train_aug --train_n_way 20 --n_shot 5 --output_dim 512
    # python train.py --dataset miniImagenet --model ResNet34 --method baseline --train_aug --train_n_way 20 --n_shot 5 --output_dim 1024
    # python train.py --dataset miniImagenet --model ResNet34 --method baseline --train_aug --train_n_way 20 --n_shot 5 --output_dim 2048

