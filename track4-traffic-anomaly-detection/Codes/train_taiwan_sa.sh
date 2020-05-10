python train_taiwan_sa.py \
    -g 0 \
    --dataset taiwan_sa \
    -b 4 \
    -i 150000 \
    --per_step 10000 \
    --num_his 4 \
    --train_folder /media/DATA/VAD_datasets/taiwan_sa/training/frames \
    --test_folder /media/DATA/VAD_datasets/taiwan_sa/testing/frames \
    --config training_hyper_params/hyper_params.ini \
    --evaluate compute_auc
    #/media/DATA/HEVI_dataset/frames \

# python train_taiwan_sa.py -g 0 --dataset taiwan_sa -b 4 -i 200000 --per_step 10000 --num_his 4 --train_folder /media/DATA/VAD_datasets/taiwan_sa/training/frames --test_folder /media/DATA/VAD_datasets/taiwan_sa/testing/frames --config training_hyper_params/hyper_params.ini --evaluate compute_auc
#python train_taiwan_sa.py -g 0 --dataset ucf_crimes -b 4 -i 200000 --per_step 10000 --num_his 4 --train_folder /media/DATA/VAD_datasets/UCF_Crimes/frames/training/ --test_folder /media/DATA/VAD_datasets/UCF_Crimes/frames/testing/ --config training_hyper_params/hyper_params.ini --evaluate compute_auc
