"""
store the arguments can be modified by the user
"""
import argparse

bert_arguments = {
    "gpu": {
        "type": "bool",
        "default": True,
        "help": "whether use gpu"
    },
    "gpu_id": {
        "type": "int",
        "default": None,
        "help": "the gpu id to use"
    },
    "seed": {
        "type": "int",
        "default": None,
        "help": "random seed"
    },
    "train_rate": {
        "type": "float",
        "default": None,
        "help": "the train set rate"
    },
    "eval_rate": {
        "type": "float",
        "default": None,
        "help": "the validation set rate"
    },
    "batch_size": {
        "type": "int",
        "default": None,
        "help": "the batch size"
    },
    "learning_rate": {
        "type": "float",
        "default": None,
        "help": "learning rate"
    },
    "max_epoch": {
        "type": "int",
        "default": None,
        "help": "the maximum epoch"
    },
    "train": {
        "type": "bool",
        "default": True,
        "help": "whether re-train model if the model is trained before"
    },
    "saved_model": {
        "type": "bool",
        "default": True,
        "help": "whether save the trained model"
    },
    "dataset_class": {
        "type": "str",
        "default": None,
        "help": "the dataset class name"
    },
    "executor": {
        "type": "str",
        "default": None,
        "help": "the executor class name"
    },
    "evaluator": {
        "type": "str",
        "default": None,
        "help": "the evaluator class name"
    },
    "roadmap_path": {
        "type": "str",
        "default": None,
        "help": "roadmap path"
    },
    "vocab_path": {
        "type": "str",
        "default": None,
        "help": "built vocab model path with bert-vocab"
    },
    "min_freq": {
        "type": "int",
        "default": None,
        "help": "Minimum frequency of occurrence of road segments"
    },
    "merge": {
        "type": "bool",
        "default": None,
        "help": "Whether to merge 3 dataset to get vocab"
    },
    "d_model": {
        "type": "int",
        "default": None,
        "help": "hidden size of transformer model"
    },
    "mlp_ratio": {
        "type": "int",
        "default": None,
        "help": "The ratio of FNN layer dimension to d_model"
    },
    "n_layers": {
        "type": "int",
        "default": None,
        "help": "number of layers"
    },
    "attn_heads": {
        "type": "int",
        "default": None,
        "help": "number of attention heads"
    },
    "seq_len": {
        "type": "int",
        "default": None,
        "help": "maximum sequence len"
    },
    "future_mask": {
        "type": "bool",
        "default": None,
        "help": "Whether to mask the future timestep, True is single-direction attention, False for double-direction"
    },
    "dropout": {
        "type": "float",
        "default": None,
        "help": " The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
    },
    "attn_drop": {
        "type": "float",
        "default": None,
        "help": "The dropout ratio for the attention probabilities."
    },
    "drop_path": {
        "type": "float",
        "default": None,
        "help": "dropout of encoder block"
    },
    "masking_ratio": {
        "type": "float",
        "default": None,
        "help": "mask ratio of input"
    },
    "masking_mode": {
        "type": "str",
        "default": None,
        "help": "mask all dim together or mask dim separate"
    },
    "distribution": {
        "type": "str",
        "default": None,
        "help": "random mask or makov mask i.e. independent mask or continuous mask"
    },
    "avg_mask_len": {
        "type": "int",
        "default": None,
        "help": "average mask length for makov mask i.e. continuous mask"
    },
    "type_ln": {
        "type": "str",
        "default": None,
        "help": "pre-norm or post-norm"
    },
    "test_every": {
        "type": "int",
        "default": None,
        "help": "Frequency of testing on the test set"
    },
    "learner": {
        "type": "str",
        "default": None,
        "help": "type of optimizer"
    },
    "grad_accmu_steps": {
        "type": "int",
        "default": None,
        "help": "learning rate"
    },
    "lr_decay": {
        "type": "bool",
        "default": None,
        "help": "whether to use le_scheduler"
    },
    "lr_scheduler": {
        "type": "str",
        "default": None,
        "help": "type of lr_scheduler"
    },
    "lr_eta_min": {
        "type": "float",
        "default": None,
        "help": "min learning rate"
    },
    "lr_warmup_epoch": {
        "type": "int",
        "default": None,
        "help": "warm-up epochs"
    },
    "lr_warmup_init": {
        "type": "float",
        "default": None,
        "help": "initial lr for warm-up"
    },
    "t_in_epochs": {
        "type": "bool",
        "default": None,
        "help": "whether update lr epoch by epoch(True) / batch by batch(False)"
    },
    "clip_grad_norm": {
        "type": "bool",
        "default": None,
        "help": "Whether to use gradient cropping"
    },
    "max_grad_norm": {
        "type": "float",
        "default": None,
        "help": "Maximum gradient"
    },
    "use_early_stop": {
        "type": "bool",
        "default": None,
        "help": "Whether to use early-stop"
    },
    "patience": {
        "type": "int",
        "default": None,
        "help": "early-stop epochs"
    },
    "log_every": {
        "type": "int",
        "default": None,
        "help": "Frequency of logging epoch by epoch"
    },
    "log_batch": {
        "type": "int",
        "default": None,
        "help": "Frequency of logging batch by batch"
    },
    "load_best_epoch": {
        "type": "bool",
        "default": None,
        "help": "Whether to load best model for test"
    },
    "l2_reg": {
        "type": "bool",
        "default": None,
        "help": "Whether to use L2 regularization"
    },
    "initial_ckpt": {
        "type": "str",
        "default": None,
        "help": "Path of the model parameters to be loaded"
    },
    "unload_param": {
        "type": "list of str",
        "default": None,
        "help": "unloaded pretrain parameters"
    },
    "add_cls": {
        "type": "bool",
        "default": None,
        "help": "Whether add CLS in BERT"
    },
    "pooling": {
        "type": "str",
        "default": None,
        "help": "Trajectory embedding pooling method"
    },
    "pretrain_path": {
        "type": "str",
        "default": None,
        "help": "Path of pretrained model"
    },
    "freeze": {
        "type": "bool",
        "default": None,
        "help": "Whether to freeze the pretrained BERT"
    },
    "topk": {
        "type": "list of int",
        "default": None,
        "help": "top-k value for classification evaluator"
    },
    "n_views": {
        "type": "int",
        "default": None,
        "help": "number of views for contrastive learning"
    },
    "similarity": {
        "type": "str",
        "default": None,
        "help": "similarity of different representations for contrastive learning"
    },
    "temperature": {
        "type": "float",
        "default": None,
        "help": "temperature of nt-xent loss for contrastive learning"
    },
    "contra_ratio": {
        "type": "float",
        "default": None,
        "help": "contrastive loss ratio"
    },
    "contra_loss_type": {
        "type": "str",
        "default": None,
        "help": "contrastive loss type, i.e. simclr, simsce, consert"
    },
    "mlm_ratio": {
        "type": "float",
        "default": None,
        "help": "mlm(predict location task) loss ratio"
    },
    "cutoff_row_rate": {
        "type": "float",
        "default": None,
        "help": "cutoff_row_rate for data argument"
    },
    "cutoff_column_rate": {
        "type": "float",
        "default": None,
        "help": "cutoff_column_rate for data argument"
    },
    "cutoff_random_rate": {
        "type": "float",
        "default": None,
        "help": "cutoff_random_rate for data argument"
    },
    "sample_rate": {
        "type": "float",
        "default": None,
        "help": "sample_rate for data argument"
    },
    "data_argument1": {
        "type": "list of str",
        "default": None,
        "help": "data argument methods for view1"
    },
    "data_argument2": {
        "type": "list of str",
        "default": None,
        "help": "data argument methods for view2"
    },
    "split": {
        "type": "bool",
        "default": None,
        "help": "split=True, load out_data_argument1/2 to split load enhanced train/eval data"
    },
    "out_data_argument1": {
        "type": "str",
        "default": None,
        "help": "data argument methods for view1"
    },
    "out_data_argument2": {
        "type": "str",
        "default": None,
        "help": "data argument methods for view2"
    },
    "classify_label": {
        "type": "str",
        "default": None,
        "help": "classify label for downstream task, vflag, usrid"
    },
    "use_pack": {
        "type": "bool",
        "default": None,
        "help": "whether use pack method in base rnn model"
    },
    "cluster_kinds": {
        "type": "int",
        "default": None,
        "help": "cluster kinds"
    },
    "add_time_in_day": {
        "type": "bool",
        "default": None,
        "help": "whether use time_in_day emb"
    },
    "add_day_in_week": {
        "type": "bool",
        "default": None,
        "help": "whether use day_in_week emb"
    },
    "add_pe": {
        "type": "bool",
        "default": None,
        "help": "whether use position emb"
    },
    "roadnetwork": {
        "type": "str",
        "default": None,
        "help": "road network dataset"
    },
    "geo_file": {
        "type": "str",
        "default": None,
        "help": "road network dataset"
    },
    "rel_file": {
        "type": "str",
        "default": None,
        "help": "road network dataset"
    },
    "bidir_adj_mx": {
        "type": "bool",
        "default": None,
        "help": "whether use bi-dir adj_mx forced"
    },
    "cluster_data_path": {
        "type": "str",
        "default": None,
        "help": "test data name for cluster"
    },
    "query_data_path": {
        "type": "str",
        "default": None,
        "help": "test data name for similarity-search"
    },
    "detour_data_path": {
        "type": "str",
        "default": None,
        "help": "test detour data name for similarity-search"
    },
    "origin_big_data_path": {
        "type": "str",
        "default": None,
        "help": "test database name of for similarity-search"
    },
    "sim_select_num": {
        "type": "int",
        "default": None,
        "help": "num of trajectories in similarity-search task"
    },
    "add_temporal_bias": {
        "type": "bool",
        "default": None,
        "help": "whether add time-aware transformer"
    },
    "temporal_bias_dim": {
        "type": "int",
        "default": None,
        "help": "hidden dim of time interval"
    },
    "model_cache_file": {
        "type": "str",
        "default": None,
        "help": "mode path to load when not train"
    },
    "add_gat": {
        "type": "bool",
        "default": None,
        "help": "whether use GAT to encode node emb"
    },
    "gat_heads_per_layer": {
        "type": "list of int",
        "default": None,
        "help": "GAT heads for per layer"
    },
    "gat_features_per_layer": {
        "type": "list of int",
        "default": None,
        "help": "GAT features for per layer per head"
    },
    "gat_dropout": {
        "type": "float",
        "default": None,
        "help": "dropout of GAT"
    },
    "gat_K": {
        "type": "int",
        "default": None,
        "help": "K-order neighbors, K-step transfer probability"
    },
    "gat_avg_last": {
        "type": "bool",
        "default": None,
        "help": "whether avg heads of GAT last layer"
    },
    "load_trans_prob": {
        "type": "bool",
        "default": None,
        "help": "whether use location transfer prob in GAT"
    },
    "sim_mode": {
        "type": "str",
        "default": None,
        "help": "similarity evaluator mode, most similarity or knn similarity"
    },
    "max_train_size": {
        "type": "int",
        "default": None,
        "help": "max number of trajectory in train dataset"
    }
}


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('bool value expected.')


def str2float(s):
    if isinstance(s, float):
        return s
    try:
        x = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError('float value expected.')
    return x


def add_other_args(parser):
    data = bert_arguments
    for arg in data:
        if data[arg]['type'] == 'int':
            parser.add_argument('--{}'.format(arg), type=int,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'bool':
            parser.add_argument('--{}'.format(arg), type=str2bool,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'float':
            parser.add_argument('--{}'.format(arg), type=str2float,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'str':
            parser.add_argument('--{}'.format(arg), type=str,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'list of int':
            parser.add_argument('--{}'.format(arg), nargs='+', type=int,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'list of str':
            parser.add_argument('--{}'.format(arg), nargs='+', type=str,
                                default=data[arg]['default'], help=data[arg]['help'])
