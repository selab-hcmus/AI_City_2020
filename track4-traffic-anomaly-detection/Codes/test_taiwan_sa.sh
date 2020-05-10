python test_taiwan_sa.py -g 0 --dataset taiwan_sa -b 4 -i 1 --num_his 4 --test_folder /media/DATA/VAD_datasets/taiwan_sa/testing/frames/ --config training_hyper_params/hyper_params.ini --evaluate compute_auc

#python test_taiwan_sa.py -g 0 --dataset A3D -b 4 -i 1 --num_his 4 --test_folder /media/DATA/A3D/frames/ --config training_hyper_params/hyper_params.ini --evaluate compute_auc

#python test_taiwan_sa.py -g 0 --dataset ucf_crimes -b 4 -i 1 --num_his 4 --test_folder /media/DATA/VAD_datasets/UCF_Crimes/frames/testing/ --config training_hyper_params/hyper_params.ini --evaluate compute_auc

#python test_taiwan_sa.py -g 0 --dataset taiwan_sa -b 4 -i 1 --num_his 4 --snapshot_dir checkpoints/taiwan_sa_l_2_alpha_1_lp_1.0_adv_0.05_gdl_1.0_flow_2.0/model.ckpt-200000 --test_folder /media/DATA/VAD_datasets/taiwan_sa/testing/frames/ --config training_hyper_params/hyper_params.ini --evaluate compute_auc

#python test_taiwan_sa.py -g 0 --dataset ucf_crimes -b 4 -i 1 --num_his 4 --snapshot_dir checkpoints/ucf_crimes_l_2_alpha_1_lp_1.0_adv_0.05_gdl_1.0_flow_2.0/model.ckpt-20000 --test_folder /media/DATA/VAD_datasets/UCF_Crimes/frames/testing/ --config training_hyper_params/hyper_params.ini --evaluate compute_auc
