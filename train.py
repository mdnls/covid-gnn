import argparse
import logging
import yaml
import os
import shutil
import torch
    # parse config file





def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True,  help='Path to config file')
    parser.add_argument('--name', type=str, required=True, help='Name of this experiment')
    parser.add_argument('--overwrite', action="store_true", help='Force overwrite any experiment output')

    args = parser.parse_args()

    with open(os.path.join('configs', args.config), 'r') as f:
        config = dict2namespace(yaml.load(f))

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    config.device = device

    exp_dir = os.path.join("experiments", args.name)

    if(not os.path.exists(exp_dir)):
        os.makedirs(exp_dir)
    else:
        overwrite = args.overwrite
        if(not overwrite):
            response = input("Experiment folder already exists. Overwrite? (Y/N)")
            overwrite = (response == "Y")

        if(overwrite):
            shutil.rmtree(exp_dir)
            os.makedirs(exp_dir)

    config.exp_dir = exp_dir
    config.name = args.name

    # TODO: add cmd argument to simulate() or train() the model. Use if statement to call appropriate placeholder methods in controller

