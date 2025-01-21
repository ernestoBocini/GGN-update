# -*- coding: utf-8 -*

import random
import collections
import time
from os import walk
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import confusion_matrix

from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch import nn

from models.ggn import GGN
from eeg_util import *
from data_util import get_dataset_info
import eeg_util
from models.baseline_models import *


import networkx as nx
import mne

import pickle
import json

import wandb

import re

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_mantis_data(args, feature_name=""):
    """
    Load data for LOOCV or default train-test split.
    """
    if args.testing_type == 'loocv':
        # Load subject-wise data for LOOCV
        subj_dir = os.path.join(args.data_path, feature_name)
        subjects = sorted(os.listdir(subj_dir))
        subject_data = {}

        for subj in subjects:
            x_path = os.path.join(subj_dir, subj, f"mwl_x_{feature_name}.npy")
            y_path = os.path.join(subj_dir, subj, f"mwl_y_{feature_name}.npy")
            subject_data[subj] = {
                "x": np.load(x_path),
                "y": np.load(y_path)
            }
        return subject_data
    else:
        # Default train-test split
        feature = np.load(os.path.join(args.data_path, f"mwl_x_{feature_name}.npy"))
        label = np.load(os.path.join(args.data_path, f"mwl_y_{feature_name}.npy"))

        label_dict = {}
        for i, l in enumerate(label):
            if l not in label_dict:
                label_dict[l] = []
            label_dict[l].append(i)

        # Split train/test
        train_x, train_y, test_x, test_y = [], [], [], []
        for k, v in label_dict.items():
            test_size = int(len(v) / 6)
            train_x.append(feature[v[test_size:]])
            train_y.append(label[v[test_size:]])
            test_x.append(feature[v[:test_size]])
            test_y.append(label[v[:test_size]])
        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y)
        test_x = np.concatenate(test_x)
        test_y = np.concatenate(test_y)

        # Reshape
        train_x = train_x.transpose(0, 3, 2, 1)
        test_x = test_x.transpose(0, 3, 2, 1)

        return [train_x, test_x], [train_y, test_y]



def generate_tuh_data(args, file_name=""):
    """ generate data for training or plotting functional connectivity.
    """
    data_path = args.data_path

    freqs = [12]
    x_data = []
    y_data = []

    types_dict = {}
    for freq in freqs:
        x_f_data = []
        y_f_data = []
        min_len = 10000
        freq_file_name = f"fft_seizures_wl1_ws_0.25_sf_250_fft_min_1_fft_max_{freq}"
        print(os.path.join(data_path, freq_file_name))
        dir, _, files = next(walk(os.path.join(data_path, freq_file_name)))
        for i, name in enumerate(files):
            fft_data = pickle.load(open(os.path.join(dir,name), 'rb'))
            if fft_data.seizure_type.upper() == 'MYSZ':
                continue
            if fft_data.seizure_type.upper() == 'BCKG':
                continue
            if fft_data.data.shape[0] < 34:
                continue
            if fft_data.data.shape[0] < min_len:
                min_len = fft_data.data.shape[0]
                
            x_f_data.append(fft_data.data)
            y_f_data.append(label_dict[fft_data.seizure_type.upper()])
        print('min len:', min_len)
        x_f_data = [d[:min_len,...] for d in x_f_data]
        x_f_data = np.stack(x_f_data, axis=0)
        print(x_f_data.shape)
        y_f_data = np.stack(y_f_data, axis=0)
        print(y_f_data.shape)
        x_data.append(x_f_data)
        y_data.append(y_f_data)

    # check each y_f_data:
    print('prepare save!')
    x_data = np.concatenate(x_data, axis=3)
    print('x data shape:', x_data.shape)
    np.save(f'seizure_x_{file_name}.npy', x_data)
    np.save(f'seizure_y_{file_name}.npy', y_data[0])
    print('y data shape:', y_data[0].shape)
    print('save done!')
    
def generate_mantis_data(args, file_name="mantis"):
    """
    Generate data for training or plotting functional connectivity, 
    similar to generate_tuh_data, but adapted for MANTIS data.
    """
    
    data_path = args.data_path
    
    x_data = []
    y_data = []
    
    x_f_data = []
    y_f_data = []
    min_len = 10000
    max_len = 0
    
    subjects = os.listdir(data_path)
    subjects.sort()
    
    for subj in tqdm(subjects, desc="Processing subjects"):
        sessions = os.listdir(os.path.join(data_path, subj))
        sessions.sort()
        
        for se in sessions:
            labels_df = pd.read_csv(os.path.join(data_path, subj, se, 'conditions.csv'))
            fif_data_path = os.path.join(data_path, subj, se, 'samples_epo.fif')
            epoched_data = mne.read_epochs(fif_data_path, verbose='CRITICAL').get_data()
            epoched_data = np.fft.rfft(epoched_data, axis=-1)
            
            if epoched_data.shape[0] < 500:
                continue
            if epoched_data.shape[0] < min_len:
                min_len = epoched_data.shape[0]
            if epoched_data.shape[0]> max_len:
                max_len = epoched_data.shape[0]
                
            x_f_data.append(epoched_data)
            y_f_data.append(np.array([label_dict[l] for l in labels_df['label']]))
    x_f_data = [d[:min_len,...] for d in x_f_data]
    y_f_data = [d[:min_len,...] for d in y_f_data]
    x_f_data = np.stack(x_f_data, axis=0)
    y_f_data = np.stack(y_f_data, axis=0)
    x_data.append(x_f_data)
    y_data.append(y_f_data)
    
    print('prepare save!')
    print('min_len: ', min_len)
    print('max_len: ', max_len)
    x_data = np.concatenate(x_data, axis=3)
    print('x data shape:', x_data.shape)
    np.save(f'seizure_x_{file_name}.npy', x_data)
    np.save(f'seizure_y_{file_name}.npy', y_data[0])
    print('y data shape:', y_data[0].shape)
    print('save done!')


def normalize_seizure_features(features):
    """inplace-norm
    Args:
        features (list of tensors): train,test,val
    """
    for i in range(len(features)):
        # (B, F, N, T)
        for j in range(features[i].shape[-1]):
            features[i][..., j] = normalize(features[i][..., j])
    
def generate_dataloader_seizure(features, labels, args):
    """
     features: [train, test, val], if val is empty then val == test
     train: B, T, N, F(12,24,48,64,96)
    """
    cates = ['train', 'test', 'val']
    datasets = dict()
    # normalize over feature dimension

    for i in range(len(features)):
        if cates[i] == 'test':
            datasets[cates[i] + '_loader'] = SeqDataLoader(features[i], labels[i], args.batch_size, cuda=args.cuda, pad_with_last_sample=False) #, drop_last=True)
            
        else:
            datasets[cates[i] + '_loader'] = SeqDataLoader(features[i], labels[i], args.batch_size, cuda=args.cuda, pad_with_last_sample=True) #, drop_last=False)

    if len(features) < 3: # take test as validation.
        datasets['val_loader'] = SeqDataLoader(features[-1], labels[-1], args.batch_size, cuda=args.cuda)

    return datasets



def init_adjs(args, index=0):
    adjs = []
   
    if args.adj_type == 'rand10':
        print('WARNING: creating adjecency matrix at random! Using 31 channels, change this for the actual run!!!!')
        adj_mx = eeg_util.generate_rand_adj(0.1*(index+1), N=31)
    elif args.adj_type == 'er':
        print('WARNING: creating adjecency matrix using ER! Using 31 channels, change this for the actual run!!!!')
        adj_mx = nx.to_numpy_array(nx.erdos_renyi_graph(31, 0.1*(index+1)))
    else:
        adj_mx = load_eeg_adj(args.adj_file, args.adj_type)
    adjs.append(adj_mx)

    #     model = EEGEncoder(adj_mx, args, is_gpu=args.cuda)
    adj = torch.from_numpy(adjs[0]).float().cuda()
    adjs[0] = adj
    return adjs

def check_trainable_parameters(model):
    """
    Prints and counts trainable vs non-trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Non-Trainable Parameters: {non_trainable_params}")
    return trainable_params, non_trainable_params



def chose_model(args, adjs):
    if args.task.upper() == 'GGN':
        adj = adjs[0]
        model = GGN(adj, args)

    elif args.task == 'transformer':
        DEVICE = torch.device("cuda:0" if args.cuda else "cpu")  
        print(f'use device: {DEVICE}')
        models = 512
        hiddens = 1024
        q = 8
        v = 8
        h = 8
        N = 8
        dropout = 0.2
        pe = True  # # 设置的是双塔中 score=pe score=channel默认没有pe
        mask = True  # 设置的是双塔中 score=input的mask score=channel默认没有mask

        inputs = 40
        channels = 14
        outputs = args.predict_class_num  # 分类类别
        hz = args.feature_len
        model = Transformer(d_model=models, d_input=inputs, d_channel=channels, d_hz = hz, d_output=outputs, d_hidden=hiddens,
                        q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE)

    elif args.task == 'gnnnet':
        model = DCRNNModel_classification(
        args, adjs, adjs[0].shape[0], args.predict_class_num, args.feature_len, device='cuda')
    elif args.task == 'cnnnet':
        model = CNNNet(args)
    else:
        model = None
        print('No model found!!!!')
    return model



def init_trainer(model, mwl_levels, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Log optimizer state
    print("New Optimizer Initialized. State should be empty:", optimizer.state_dict()['state'])

    def lr_adjust(epoch):
        if epoch < 20:
            return 1
        return args.lr_decay_rate ** ((epoch - 19) / 3 + 1)

    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_adjust)

    # Log initial learning rate
    for param_group in optimizer.param_groups:
        print(f"Initial learning rate: {param_group['lr']}")

    # Criterion setup
    c = mwl_levels
    w = np.array([c[i] for i in c.keys()])
    m = np.median(w)
    total = np.sum(w)
    weights = None
    if args.weighted_ce == 'prop':
        weights = 1 - w / total
    elif args.weighted_ce == 'rand':
        weights = np.random.rand(7) * 10
    elif args.weighted_ce == 'median':
        weights = m / w
    if weights is not None:
        weights = torch.from_numpy(weights).float().cuda()
    print('Loss weights:', weights)

    crite = FocalLoss(nn.CrossEntropyLoss(weight=weights), alpha=0.9, gamma=args.focal_gamma) if args.focalloss else nn.CrossEntropyLoss(weight=weights)

    trainer = Trainer(args, model, optimizer, criterion=crite, sched=lr_sched)

    # Log trainer initialization
    print("Trainer initialized with model, optimizer, scheduler, and criterion.")
    return trainer


def train_eeg(args, datasets, mwl_levels, subject, index=0):
    # SummaryWriter
    window_len = int(re.findall("\d+", args.data_path)[0])
    wandb.init(
        project=f"mwl-{args.dataset}-{args.task}-{args.encoder}-{window_len}",  # Set your wandb project name
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "model": args.task,
            "dataset": args.dataset,
            "seed": args.seed,
            "fold": index
        },
        name = subject
    )


    import os
    dt = time.strftime("%m_%d_%H_%M", time.localtime())
    log_dir = "./tfboard/"+args.server_tag+"/" + dt
    print('tensorboard path:', log_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)
    
    adjs = init_adjs(args, index)
    model = chose_model(args, adjs)
    
    wandb.watch(model, log="all", log_freq=10)
    
    check_trainable_parameters(model)
    

    print('args_cuda:', args.cuda)
    if args.cuda:
        print('rnn_train RNNBlock to cuda!')
        model.cuda()
    else:
        print('rnn_train RNNBlock to cpu!')

    # add scheduler.
    trainer = init_trainer(model, mwl_levels, args)
    
    best_val_loss= np.inf
    best_unchanged_threshold = 100  # accumulated epochs of best val_mae unchanged
    best_count = 0
    best_index = -1
    train_val_metrics = []
    start_time = time.time()
    basedir, file_tag = os.path.split(args.best_model_save_path)
    os.makedirs(basedir, exist_ok=True)
    model_save_path = os.path.join(basedir, f'{index}_{file_tag}')
    
    for e in range(args.epochs):
        datasets['train_loader'].shuffle()
        train_loss, train_preds = [], []

        for input_data, target in datasets['train_loader'].get_iterator():
            loss, preds = trainer.train(input_data, target)
            # training metrics
            train_loss.append(loss)
            train_preds.append(preds)
        # validation metrics
        val_loss, val_preds = [], []
    
        for input_data, target in datasets['val_loader'].get_iterator():
            loss, preds  = trainer.eval(input_data, target)
            # add metrics
            val_loss.append(loss)
            val_preds.append(preds)

        # cal metrics as a whole:
        train_preds = torch.cat(train_preds, dim=0)
        val_preds = torch.cat(val_preds, dim=0)
        
        train_acc = eeg_util.calc_eeg_accuracy(train_preds, datasets['train_loader'].ys)
        val_acc = eeg_util.calc_eeg_accuracy(val_preds, datasets['val_loader'].ys)
        
        train_loss = np.mean(train_loss)
        val_loss = np.mean(val_loss)
        
        wandb.log({
            "epoch": e,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })



        m = dict(train_loss=train_loss, train_acc=train_acc,
                 val_loss=val_loss, val_acc=val_acc)

        m = pd.Series(m)

        if e % 10 == 0:
            print('epoch:', e)
            print(m)
        # write to tensorboard:
        writer.add_scalars(f'epoch/loss', {'train': m['train_loss'], 'val': m['val_loss']}, e)
        writer.add_scalars(f'epoch/acc', {'train': m['train_acc'], 'val': m['val_acc']}, e)

        train_val_metrics.append(m)
        if m['val_loss'] <= best_val_loss:
            best_val_loss = m['val_loss']
            best_count = 0
            print("update best model, epoch: ", e)
            torch.save(trainer.model.state_dict(), model_save_path)
            print(m)
            best_index = e
        else:
            best_count += 1
        if best_count > best_unchanged_threshold:
            print('Got best')
            break

        trainer.lr_schedule()
    print('training: :')
    if args.lgg:
        print('after training adj_fix', trainer.model.LGG.adj_fix[0])
    print('best_epoch:', best_index)

    test_model = chose_model(args, adjs)
    test_model.cuda()
    with torch.no_grad():
        dummy_input = torch.randn(1, 126, 31, 30).cuda() # torch.randn(1, 126, 31, 30).cuda()
        _ = test_model(dummy_input)

    test_model.load_state_dict(torch.load(model_save_path), strict=False)

    trainer.model = test_model
    if args.lgg:
        print('after load best model adj_fix', trainer.model.LGG.adj_fix[0])
    
    test_metrics = []
    test_loss, test_preds = [], []
    

    for input_data, target in datasets['test_loader'].get_iterator():
        loss, preds = trainer.eval(input_data, target)
        
        if preds.dim() == 1:
            preds = preds.unsqueeze(0)
            
        # add metrics
        test_loss.append(loss)
        test_preds.append(preds)
        
    # cal metrics as a whole:

    # reshape:
    # Just before concatenating test_preds
    test_preds = torch.cat(test_preds, dim=0)
    test_preds = torch.softmax(test_preds, dim=1)
    print('Test preds dim: ', len(test_preds))
    print('Test loader dim: ', len(datasets['test_loader'].ys))
    test_acc = eeg_util.calc_eeg_accuracy(test_preds, datasets['test_loader'].ys)
    
    test_loss = np.mean(test_loss)
    
    wandb.log({"test_loss": test_loss, "test_acc": test_acc})
    wandb.finish()

    m = dict(test_acc=test_acc, test_loss=test_loss)
    m = pd.Series(m)
    print("test:")
    print(m)
    test_metrics.append(m)
    preds_b = test_preds.argmax(dim=1)
    
    basedir, file_tag = os.path.split(args.fig_filename)
    date_dir = time.strftime('%Y%m%d', time.localtime(time.time()))
    fig_save_dir = os.path.join(basedir, date_dir)
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)
    confused_fig_dir = os.path.join(fig_save_dir, f'{file_tag}_{index}_confusion.png')
    loss_fig_dir = os.path.join(basedir, date_dir, f'{file_tag}_{index}_loss.png')
    
    plot_confused_cal_f1(preds_b, datasets['test_loader'].ys, fig_dir=confused_fig_dir)
    plot(train_val_metrics, test_metrics, loss_fig_dir)

    print('finish rnn_train!, time cost:', time.time() - start_time)
    return train_val_metrics, test_metrics


def cal_f1(preds, labels):

    mi_f1 = f1_score(labels, preds, average='micro')
    ma_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')


    return mi_f1, ma_f1, weighted_f1

def plot_confused_cal_f1(preds, labels, fig_dir):
    preds = preds.cpu()
    labels = labels.cpu()
    
    ori_preds = preds
    sns.set()
    fig = plt.figure(figsize=(5, 4), dpi=100)
    ax = fig.gca()
    gts = [number_label_dict[int(l)][:-2] for l in labels]
    preds = [number_label_dict[int(l)][:-2] for l in preds]
    
    label_names = [v[:-2] for v in number_label_dict.values()]
    print(label_names)
    C2= np.around(confusion_matrix(gts, preds, labels=label_names, normalize='true'), decimals=2)

    # from confusion to ACC, micro-F1, macro-F1, weighted-f1.
    print('Confusion:', C2)
    mi_f1, ma_f1, w_f1 = cal_f1(ori_preds, labels)
    print(f'micro f1: {mi_f1}, macro f1: {ma_f1}, weighted f1: {w_f1}')

    sns.heatmap(C2, cbar=True, annot=True, ax=ax, cmap="YlGnBu", square=True,annot_kws={"size":9},
        yticklabels=label_names,xticklabels=label_names)

    ax.figure.savefig(fig_dir, transparent=False, bbox_inches='tight')


def plot(train_val_metrics, test_metrics, fig_filename='mae'):
    epochs = len(train_val_metrics)
    x = range(epochs)
    train_loss = [m['train_loss'] for m in train_val_metrics]
    val_loss = [m['val_loss'] for m in train_val_metrics]

    plt.figure(figsize=(8, 6))
    plt.plot(x, train_loss, '', label='train_loss')
    plt.plot(x, val_loss, '', label='val_loss')
    plt.title('loss')
    plt.legend(loc='upper right')  # 设置label标记的显示位置
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_filename)
    

import pickle

def save_results(results, file_path):
    """Save results to a JSON or pickle file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {file_path}")


def multi_train(args, feature_name='_nback_easy_v_med_v_hard', mwl_levels={'easy': 1049, 'medium': 1048, 'hard': 999}, tags="", runs=10):
    """
    Train multiple times or perform LOOCV based on args.testing.
    Use separate validation subjects in addition to test subjects.
    Save intermediate LOOCV results to a file after each fold.
    """
    results_path = os.path.join(args.data_path, 'results', f"{window_len}",  f"{feature_name}", f"{args.task}", f"{feature_name}_loocv_results.pkl")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Check if results already exist
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            existing_results = pickle.load(f)
            print("Found existing results. Continuing from the last completed fold.")
            subject_wise_results = existing_results.get("subject_wise_results", [])
            completed_subjects = {res["test_subject"] for res in subject_wise_results}
            overall_metrics = existing_results.get("overall_metrics", {})
    else:
        subject_wise_results = []
        completed_subjects = set()
        overall_metrics = {}

    if args.testing_type == 'loocv':
        subject_data = load_mantis_data(args, feature_name=feature_name)
        subjects = list(subject_data.keys())
        test_results = []
        train_results = []
        val_results = []
        time_per_fold = []

        for i, test_subj in enumerate(subjects):
            if test_subj in completed_subjects:
                print(f"Skipping subject {test_subj} (already completed).")
                continue
        
            if int(test_subj) in [4, 29, 32, 36, 43, 48]:
                continue
            
            print(f"LOOCV: Testing on subject {test_subj}, training and validating on the rest.")

            train_x, train_y, val_x, val_y = [], [], [], []
            test_x, test_y = subject_data[test_subj]['x'].copy(), subject_data[test_subj]['y'].copy()
            # Exclude test subject and randomly pick validation subjects
            remaining_subjects = [subj for subj in subjects if subj != test_subj]
            val_subjects = random.sample(remaining_subjects, min(5, len(remaining_subjects)))
            for subj in remaining_subjects:
                if subj in val_subjects:
                    val_x.append(subject_data[subj]['x'].copy())
                    val_y.append(subject_data[subj]['y'].copy())
                else:
                    train_x.append(subject_data[subj]['x'].copy())
                    train_y.append(subject_data[subj]['y'].copy())
                    
            train_x = np.concatenate(train_x)
            train_y = np.concatenate(train_y)
            val_x = np.concatenate(val_x)
            val_y = np.concatenate(val_y)

            # Transpose the data to the expected shape (B, C, N, T)
            train_x = train_x.transpose(0, 3, 2, 1)
            val_x = val_x.transpose(0, 3, 2, 1)
            test_x = test_x.transpose(0, 3, 2, 1)
            
            # Normalize features
            normalize_seizure_features([train_x, val_x, test_x])
            # Generate dataloaders
            datasets = generate_dataloader_seizure([train_x, test_x, val_x], [train_y, test_y, val_y], args)
            
            # Train and evaluate
            start_time = time.time()
            tr_metrics, te_metrics = train_eeg(args, datasets, mwl_levels, subject=test_subj, index=i)
            elapsed_time = time.time() - start_time
            test_results.append(te_metrics[0]['test_acc'])
            train_results.append(tr_metrics[-1]['train_acc'])
            val_results.append(tr_metrics[-1]['val_acc'])
            time_per_fold.append(elapsed_time)

            # Store results with test subject information
            fold_result = {
                "test_subject": test_subj,
                "test_acc": te_metrics[0]['test_acc'],
                "train_acc": tr_metrics[-1]['train_acc'],
                "val_acc": tr_metrics[-1]['val_acc'],
                "training_time": elapsed_time
            }
            subject_wise_results.append(fold_result)

            # Save intermediate results after each fold
            save_results({
                "subject_wise_results": subject_wise_results,
                "overall_metrics": {
                    "mean_train_acc": np.mean(train_results),
                    "std_train_acc": np.std(train_results),
                    "mean_val_acc": np.mean(val_results),
                    "std_val_acc": np.std(val_results),
                    "mean_test_acc": np.mean(test_results),
                    "std_test_acc": np.std(test_results),
                    "mean_train_time": np.mean(time_per_fold),
                    "std_train_time": np.std(time_per_fold),
                }
            }, results_path)

        # Final report
        print("LOOCV Results:")
        print(f"Test Accuracies: {test_results}")
        print(f"Train Accuracies: {train_results}")
        print(f"Validation Accuracies: {val_results}")
        print(f"Mean Test Accuracy: {np.mean(test_results):.4f}, Std Dev: {np.std(test_results):.4f}")
        print(f"Mean Training Time per Fold: {np.mean(time_per_fold):.4f}s, Std Dev: {np.std(time_per_fold):.4f}s")
    else:

        # Default multiple runs
        test_loss, test_acc = [], []
        xs, ys = load_mantis_data(args, feature_name=feature_name)
        normalize_seizure_features(xs)
        datasets = generate_dataloader_seizure(xs, ys, args)

        for i in range(runs):
            tr, te = train_eeg(args, datasets, mwl_levels, i)
            test_loss.append(te[0]['test_loss'])
            test_acc.append(te[0]['test_acc'])

        # Analysis
        test_loss_m = np.mean(test_loss)
        test_loss_v = np.std(test_loss)
        test_acc_m = np.mean(test_acc)
        test_acc_v = np.std(test_acc)

        print('%s,trials: %s, t loss mean/std: %f/%f, t acc mean/std: %f%s/%f \n' % (
            tags, runs, test_loss_m, test_loss_v, test_acc_m, '%', test_acc_v))




def testing(args, dataloaders, test_model, batch=False):
    torch.cuda.empty_cache()
    test_model.cuda()
    preds = []
    for x, y in dataloaders['test_loader'].get_iterator():
        p = test_model(x)
        preds.append(p.detach().cpu())
        del p
        torch.cuda.empty_cache()

    preds = torch.cat(preds, dim=0)
        
    preds = torch.softmax(preds, dim=1)
    
    basedir, file_tag = os.path.split(args.fig_filename)
    date_dir = time.strftime('%Y%m%d', time.localtime(time.time()))
    fig_save_dir = os.path.join(basedir, date_dir)
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)
    confused_fig_dir = os.path.join(fig_save_dir, f'testing_confusion_map_{file_tag}.png')
    
    preds_b = preds.argmax(dim=1)
    plot_confused_cal_f1(preds_b, datasets['test_loader'].ys, fig_dir=confused_fig_dir)
    
    return preds


if __name__ == "__main__":
    
    start_t = time.time()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    args = eeg_util.get_common_args()
    args = args.parse_args()
    eeg_util.DLog.init(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    args = get_common_args().parse_args()
    
    window_len = int(re.findall("\d+", args.data_path)[0])
    mwl_levels, label_dict, number_label_dict, feature_name = get_dataset_info(dataset = args.dataset, window_len=window_len)
        

    if args.testing:
        print('Unit_test!!!!!!!!!!!!!')
        if args.arg_file != 'None':
            args_dict = vars(args)
            print(args_dict.keys())
            print('testing args:')
            with open(args.arg_file, 'rt') as f:
                args_dict.update(json.load(f))
            print('args_dict keys after update:', args_dict.keys())
            args.testing = True
            
        xs, ys =  load_mantis_data(args)
        normalize_seizure_features(xs)
        datasets = generate_dataloader_seizure(xs,ys,args)
        adjs = init_adjs(args)
        test_model = chose_model(args, adjs)
        test_model.load_state_dict(torch.load(args.best_model_save_path), strict=False)
        test_model.cuda()
        test_model.eval()
        DLog.log('args is : by DLOG:', args)
        testing(args, datasets, test_model)
        
    elif args.task == 'generate_data':
        if args.dataset == 'TUH':
            generate_tuh_data(args, file_name="")
        elif args.dataset == 'MANTIS':
            generate_mantis_data(args, file_name="mantis")
    else:
        dt = time.strftime('%Y%m%d', time.localtime(time.time()))
        model_used = "basic model"
        
        tags = "type:" + model_used + str(dt)
        # Save the args:
        _, file_tag = os.path.split(args.fig_filename)
        args_path = f'./args/{dt}/'
        if not os.path.exists(args_path):
            os.makedirs(args_path)
        with open(os.path.join(args_path, f'{file_tag}.json'), 'wt') as f:
            json.dump(vars(args), f, indent=4)
            
        DLog.log('------------ Args Saved! -------------')
        DLog.log('args is : by DLOG:', args)
        multi_train(args, feature_name=feature_name, tags=tags, mwl_levels=mwl_levels, runs=args.runs)
        
    print('Main running Over, total time spent:',time.time() - start_t)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

