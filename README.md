GGN is a generative deep learning model for epilepsy seizure classification and detecting the disruptive functional connectiviy when seizure attacks.

--- 

## Preparation for dataset. 


1. Access dataset TUSZ v1.5.2 from https://isip.piconepress.com/projects/tuh_eeg/
2. Preprocess the raw data following the benchmark setting from IBM: https://github.com/IBM/seizure-type-classification-tuh
3. Composite features from different frequencies following our paper.

---



## Testing via trained model.

the shuffled_index.npy stored the indices of training samples and testing samples of the best_model.pth.

1. config the data path and trained model path in the file test.sh.
1. `sh testing.sh`
1. use `sh testing.sh kill`, to kill the running process.

## Training GGN

1. config the data path and trained model path in the file start_train.sh
2. `sh start_train.sh`
3. or you could reset the hyperparameters in start_train.sh or just set in args, e.g.,`
sh start_train.sh data_path=xxx lr=0.00002`
1. use `sh start_train.sh kill`, to kill the running process.

## Training compared models

1. `sh start_train.sh --task=cnnnet`, training CNN based model.
1. `sh start_train.sh --task=gnnnet`, training GNN based model.
1. `sh start_train.sh --task=transformer`, training Transformer based model.


## Turn on debug log mode
to print more logs, add `--debug` in the args.