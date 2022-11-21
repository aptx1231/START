import random
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_executor, get_model, get_logger, ensure_dir, set_random_seed


def run_model(task=None, model_name=None, dataset_name=None, config_file=None,
              saved_model=True, train=True, other_args=None):
    """
    Args:
        task(str): task name
        model_name(str): model name
        dataset_name(str): dataset name
        config_file(str): config filename used to modify the pipeline's
            settings. the config file should be json.
        saved_model(bool): whether to save the model
        train(bool): whether to train the model
        other_args(dict): the rest parameter args, which will be pass to the Config
    """
    # load config
    config = ConfigParser(task, model_name, dataset_name, config_file, saved_model, train, other_args)
    exp_id = config.get('exp_id', None)
    if exp_id is None:
        # Make a new experiment ID
        exp_id = int(random.SystemRandom().random() * 1000000)
        config['exp_id'] = exp_id
    logger = get_logger(config)
    logger.info('Begin pretrain-pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(task), str(model_name), str(dataset_name), str(exp_id)))
    logger.info(config.config)
    seed = config.get('seed', 0)
    set_random_seed(seed)
    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    model_cache_file = config.get('model_cache_file', './libcity/cache/{}/model_cache/{}_{}_{}.pt'.format(
        exp_id, exp_id, model_name, dataset_name))
    model = get_model(config, data_feature)
    executor = get_executor(config, model, data_feature)

    initial_ckpt = config.get("initial_ckpt", None)
    pretrain_path = config.get("pretrain_path", None)
    if train:
        executor.train(train_data, valid_data, test_data)
        if saved_model:
            executor.save_model(model_cache_file)
        executor.evaluate(test_data)
    else:
        # assert os.path.exists(model_cache_file) or initial_ckpt is not None or pretrain_path is not None
        if initial_ckpt is None and pretrain_path is None:
            executor.load_model_state(model_cache_file)
        executor.evaluate(test_data)
