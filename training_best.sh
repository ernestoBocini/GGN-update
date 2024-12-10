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


training_tag=training_default_ggn_easy_v_medium_v_hard_best

nohup python -u $proj_path/eeg_main.py \
--seed=1992 \
--task=ggn \
--runs=3 \
--wavelets_num=31 \
--batch_size=32 \
--epochs=100 \
--weighted_ce=prop \
--lr=0.0005 \
--lr_decay_rate=0.92 \
--dropout=0.5 \
--predict_class_num=3 \
--server_tag=seizure \
--data_path=$root_path/ggn_data \
--dataset=EASY_V_MEDIUM_V_HARD \
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
--gnn_hid_dim=64 \
--gnn_out_dim=16 \
--gnn_layer_num=2 \
--max_diffusion_step=2 \
--fig_filename=$proj_path/figs/$training_tag \
--dev_size=1000 \
--weight_decay=0.0001 \
--clip=3 \
--seq_length=3 \
--predict_len=12 \
--mo=0.1 \
--gnn_name=gwn \
--gnn_fin_fout="1100,550;550,128;128,128" \
--gnn_adj_type=lgg \
--gnn_downsample_dim=16 \
--coarsen_switch=3 \
--gwn_out_features=32 \
--rnn_layer_num=2 \
--rnn_in_channel=32 \
--gcn_out_features=32 \
--rnn_hidden_len=32 \
--max_diffusion_step=2 \
--eeg_seq_len=250 \
--decoder_type=both \
--decoder_downsample=-1 \
--predictor_num=3 \
--predictor_hid_dim=512 \
--dcgru_activation=tanh \
--best_model_save_path=$proj_path/best_models/$training_tag.pth \
> $proj_path/logs/$training_tag.log 2>&1 &

echo "check log at $proj_path/logs/$training_tag.log"









