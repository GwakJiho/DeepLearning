import sys
import numpy as np

from nets.multi_layer_net import Model
from utils.trainer import Trainer
from utils.util import shuffle_dataset, load_config
from data.DataLoader import Get_Data

config = load_config("config.yaml")

x_train, t_train, x_val, t_val = Get_Data()

network = Model(input_size=config['network']['input_size'], hidden_size_list=config['network']['hidden_size'],
                                output_size=config['network']['output_size'], weight_init_std=config['network']['weight_init_std'],
                                use_batchnorm=config['network']['use_batchnorm'], weight_decay_lambda=config['network']['weight_decay'],
                                use_dropout=config['network']['use_Dropout'], dropout_ratio=config['network']['dropout_ratio'],
                                use_weights=config['network']['use_weights'], weights=config['weight']['file'])


trainer = Trainer(network, x_train, t_train, x_val, t_val,
                  epochs=200, mini_batch_size=config['train']['batch_size'],
                  optimizer=config['train']['optimizer'], optimizer_param={'lr': config['train']['lr']}, verbose=config['train']['verbose'])


trainer.train()

if config['train']['save_weight']:
    np.save(config['weight']['file'], network.params)

f = open(config['train']['log_file'], 'w')
f.write(trainer.train_log)
f.close()
...