#!/bin/bash
echo "------------ start --------"
root_path=/home/ebocini/repos/mantis_data
proj_path=/home/ebocini/repos/GGN-update

k=$1
if [ ! -n "$k" ];then
    k="train"
fi

echo $k
pid=$(ps -ef | grep seizure | grep -v grep | awk '{print $2}')
if [ -n "$pid" ]; then
    echo "running seizure: $pid" 
    kill -9 $pid
    echo "killed!"
fi

if [ $k = "kill" ]; then
    echo "only kill seizure process"
    exit 1
fi

# TUH:
echo "start running tuh eeg_train!"


training_tag=training_gnn

nohup python -u $proj_path/eeg_main.py \
--seed=1992 \
--task=gnnnet \
--runs=1 \
--wavelets_num=16 \
--batch_size=32 \
--epochs=100 \
--weighted_ce=prop \
--lr=0.0005 \
--dropout=0.3 \
--predict_class_num=3 \
--server_tag=seizure \
--data_path=$root_path/ggn_data \
--dataset=MANTIS \
--adj_file=$proj_path/adjs/raw_adj.npy \
--adj_type=er \
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
--gnn_pooling=gate \
--agg_type=gate \
--gnn_hid_dim=32 \
--gnn_out_dim=16 \
--gnn_layer_num=2 \
--max_diffusion_step=2 \
--fig_filename=$proj_path/figs/$training_tag \
--best_model_save_path=$proj_path/best_models/$training_tag.pth \
> $proj_path/logs/$training_tag.log 2>&1 &

echo "check log at $proj_path/logs/$training_tag.log"