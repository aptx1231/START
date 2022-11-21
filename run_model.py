import argparse

from libcity.pipeline import run_model
from libcity.utils import add_other_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='trajectory_embedding', help='the name of task')
    parser.add_argument('--model', type=str, default='BERTLM', help='the name of model')
    parser.add_argument('--dataset', type=str, default='porto', help='the name of dataset')
    parser.add_argument('--config_file', type=str, default=None, help='the file name of config file')
    parser.add_argument('--exp_id', type=str, default=None, help='id of experiment')
    add_other_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    other_args = {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
        val is not None}
    run_model(task=args.task, model_name=args.model, dataset_name=args.dataset,
              config_file=args.config_file, saved_model=args.saved_model,
              train=args.train, other_args=other_args)
