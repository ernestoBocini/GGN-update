#!/bin/bash

echo "------------ start --------"
root_path=/home/ebocini/repos/mantis_data
proj_path=/home/ebocini/repos/GGN-update

# Define datasets, predict_class_num, and epochs
DATASETS=("BASELINE" "EYESOPEN_V_NBACK" "EASY_V_HARD" "EASY_V_MEDIUM" "MEDIUM_V_HARD" "0_v_2" "0_v_4" "0_v_2_v_4" "1_v_3_v_5")
PREDICT_CLASS_NUM=(2 2 2 2 2 2 2 3 3)
EPOCHS=(10 20 50 50 50 50 50 50 50)

for i in "${!DATASETS[@]}"; do
    dataset=${DATASETS[$i]}
    predict_class_num=${PREDICT_CLASS_NUM[$i]}
    epochs=${EPOCHS[$i]}

    echo "Clearing CUDA memory..."
    nvidia-smi --gpu-reset -i 0 > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Failed to reset CUDA memory. Please ensure no other process is using the GPU."
    fi
    echo "CUDA memory cleared."

    echo "Running setup $((i+1)): Dataset=$dataset, Predict_Class_Num=$predict_class_num, Epochs=$epochs"

    training_tag=training_default_ggn_${dataset,,}_loocv

    python -u $proj_path/eeg_main.py \
    --seed=1992 \
    --task=ggn \
    --runs=1 \
    --wavelets_num=31 \
    --batch_size=32 \
    --epochs=$epochs \
    --weighted_ce=prop \
    --lr=0.0005 \
    --dropout=0.3 \
    --predict_class_num=$predict_class_num \
    --server_tag=seizure \
    --data_path=$root_path/ggn_data_loocv \
    --dataset=$dataset \
    --adj_file=$proj_path/adjs/A_combined.npy \
    --adj_type=origin \
    --feature_len=126 \
    --cuda \
    --encoder=rnn \
    --bidirect \
    --encoder_hid_dim=256 \
    --cut_encoder_dim=0 \
    --decoder=lgg_cnn \
    --decoder_hid_dim=512 \
    --decoder_out_dim=32 \
    --lgg_warmup=5 \
    --lgg_tau=0.01 \
    --lgg_hid_dim=64 \
    --lgg_k=5 \
    --lgg \
    --gnn_pooling=gate \
    --agg_type=gate \
    --gnn_hid_dim=32 \
    --gnn_out_dim=16 \
    --gnn_layer_num=2 \
    --max_diffusion_step=2 \
    --fig_filename=$proj_path/figs/$training_tag \
    --best_model_save_path=$proj_path/best_models/$training_tag.pth \
    > $proj_path/logs/$training_tag.log 2>&1

    echo "Check log at $proj_path/logs/$training_tag.log"
done

echo "All setups completed."
