#!/bin/bash
echo "------------ start --------"
root_path=/home/ebocini/hdd/mantis_data 
proj_path=/home/ebocini/hdd/GGN-update

k=$1
if [ ! -n "$k" ]; then
    k="train"
fi

echo $k
pid=$(ps -ef | grep mwlmantis | grep -v grep | awk '{print $2}')
if [ -n "$pid" ]; then
    echo "running mwlmantis: $pid" 
    kill -9 $pid
    echo "killed!"
fi

if [ $k = "kill" ]; then
    echo "only kill mwlmantis process"
    exit 1
fi

echo "start running tuh eeg_train!"

training_tag=training_default_ggn_BASELINE
task=ggn

# Set the environment variable to specify the GPU
export CUDA_VISIBLE_DEVICES=0

nohup python -u $proj_path/eeg_main.py \
--seed=1992 \
--task=$task \
--runs=1 \
--wavelets_num=31 \
--batch_size=256 \
--epochs=50 \
--weighted_ce=prop \
--lr=0.001 \
--dropout=0.5 \
--predict_class_num=2 \
--server_tag=mwlmantis \
--data_path=$root_path/ggn_data_loocv_30s \
--dataset=BASELINE \
--adj_file=$proj_path/adjs/A_combined_mantis_31.npy \
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
--best_model_save_path=$proj_path/best_models/$task/$training_tag.pth \
> $proj_path/logs/$training_tag.log 2>&1 &

echo "check log at $proj_path/logs/$training_tag.log"