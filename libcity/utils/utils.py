import importlib
import logging
import os
import sys
import numpy as np
import time
import random
import torch
import datetime


def get_executor(config, model, data_feature):
    """
    according the config['executor'] to create the executor

    Args:
        config(ConfigParser): config
        model(AbstractModel): model
        data_feature(dict): data_feature

    Returns:
        AbstractExecutor: the loaded executor
    """
    try:
        return getattr(importlib.import_module('libcity.executor'),
                       config['executor'])(config, model, data_feature)
    except AttributeError:
        raise AttributeError('executor is not found')


def get_model(config, data_feature):
    """
    according the config['model'] to create the model

    Args:
        config(ConfigParser): config
        data_feature(dict): feature of the data

    Returns:
        AbstractModel: the loaded model
    """
    if config['task'] == 'trajectory_embedding':
        try:
            return getattr(importlib.import_module('libcity.model.trajectory_embedding'),
                           config['model'])(config, data_feature)
        except AttributeError:
            raise AttributeError('model is not found')
    elif config['task'] == 'road_representation':
        try:
            return getattr(importlib.import_module('libcity.model.road_representation'),
                           config['model'])(config, data_feature)
        except AttributeError:
            raise AttributeError('model is not found')
    else:
        raise AttributeError('task is not found')


def get_evaluator(config, data_feature):
    """
    according the config['evaluator'] to create the evaluator

    Args:
        config(ConfigParser): config
        data_feature(dict): feature of the data

    Returns:
        AbstractEvaluator: the loaded evaluator
    """
    try:
        return getattr(importlib.import_module('libcity.evaluator'),
                       config['evaluator'])(config, data_feature)
    except AttributeError:
        raise AttributeError('evaluator is not found')


def get_local_time():
    """
    获取时间

    Return:
        datetime: 时间
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur


def get_logger(config, name=None):
    """
    获取Logger对象

    Args:
        config(ConfigParser): config
        name: specified name

    Returns:
        Logger: logger
    """
    log_dir = './libcity/log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = '{}-{}-{}-{}.log'.format(config['exp_id'],
                                            config['model'], config['dataset'], get_local_time())
    logfilepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)

    log_level = config.get('log_level', 'INFO')

    if log_level.lower() == 'info':
        level = logging.INFO
    elif log_level.lower() == 'debug':
        level = logging.DEBUG
    elif log_level.lower() == 'error':
        level = logging.ERROR
    elif log_level.lower() == 'warning':
        level = logging.WARNING
    elif log_level.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logfilepath)
    file_handler.setFormatter(formatter)

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('Log directory: %s', log_dir)
    return logger


def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def timestamp_datetime(secs):
    dt = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.localtime(secs))
    return dt


def datetime_timestamp(dt):
    s = time.mktime(time.strptime(dt, '%Y-%m-%dT%H:%M:%SZ'))
    return int(s)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
