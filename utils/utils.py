import argparse
import os
import time
import pandas as pd
import ast

import torch
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from log import logger

from model.ner_model import NERBaseAnnotator
from utils.reader import CoNLLReader
from utils.reader_utils import get_ner_reader


##2022 scheme. Retaining in case we want coarse grained else we simply overwrite
wnut_iob = {'B-CORP': 0, 'I-CORP': 1, 'B-CW': 2, 'I-CW': 3, 'B-GRP': 4, 'I-GRP': 5, 'B-LOC': 6, 'I-LOC': 7, 
            'B-PER': 8, 'I-PER': 9, 'B-PROD': 10, 'I-PROD': 11, 'O': 12}



##let us also define coarse-grained labels:
coarse_grained_tags = dict()
coarse_grained_tags['B-Product'] = ['B-OtherPROD', 'B-Drink', 'B-Food','B-Vehicle', 'B-Clothing']
coarse_grained_tags['B-Location'] = ['B-Facility', 'B-HumanSettlement','B-OtherLOC', 'B-Station']
coarse_grained_tags['B-Creative Works'] = ['B-Software', 'B-VisualWork', 'B-ArtWork', 'B-MusicalWork','B-WrittenWork']
coarse_grained_tags['B-Medical'] = ['B-MedicalProcedure', 'B-Medication/Vaccine', 'B-Symptom', 'B-AnatomicalStructure', 'B-Disease']
coarse_grained_tags['B-Person'] = ['B-SportsManager', 'B-OtherPER', 'B-Cleric','B-Scientist', 'B-Politician', 'B-Artist','B-Athlete']
coarse_grained_tags['B-Group'] = ['B-PublicCorp', 'B-SportsGRP','B-MusicalGRP', 'B-AerospaceManufacturer', 'B-CarManufacturer','B-PrivateCorp', 'B-ORG']

coarse_grained_tags['I-Product'] = ['I-OtherPROD', 'I-Drink', 'I-Food', 'I-Vehicle', 'I-Clothing']
coarse_grained_tags['I-Location'] = ['I-Facility', 'I-HumanSettlement', 'I-OtherLOC', 'I-Station']
coarse_grained_tags['I-Creative Works'] = ['I-Software', 'I-VisualWork', 'I-ArtWork', 'I-MusicalWork', 'I-WrittenWork']
coarse_grained_tags['I-Medical'] = ['I-MedicalProcedure', 'I-Medication/Vaccine', 'I-Symptom', 'I-AnatomicalStructure', 'I-Disease']
coarse_grained_tags['I-Person'] = ['I-SportsManager', 'I-OtherPER', 'I-Cleric', 'I-Scientist', 'I-Politician', 'I-Artist', 'I-Athlete']
coarse_grained_tags['I-Group'] = ['I-PublicCorp', 'I-SportsGRP', 'I-MusicalGRP', 'I-AerospaceManufacturer', 'I-CarManufacturer', 'I-PrivateCorp','I-ORG']

coarse_grained_tags['O'] = ['O']
coarse_grained_tags_dict = dict()
for k,v in coarse_grained_tags.items():
    for item in v:
        coarse_grained_tags_dict[item] = k



##Get all config related data
def parse_args():
    p = argparse.ArgumentParser(description='Model configuration.', add_help=False)
    p.add_argument('--train', type=str, help='Path to the train data.', default='en_train.conll')
    p.add_argument('--test', type=str, help='Path to the test data.', default='en_test.conll')
    p.add_argument('--dev', type=str, help='Path to the dev data.', default='en_dev.conll')
    p.add_argument('--out_dir', type=str, help='Output directory.', default='.')
    p.add_argument('--iob_tagging', type=str, help='IOB tagging scheme', default='wnut')
    p.add_argument('--max_instances', type=int, help='Maximum number of instances', default=-1)
    p.add_argument('--max_length', type=int, help='Maximum number of tokens per instance.', default=50)
    p.add_argument('--encoder_model', type=str, help='Pretrained encoder model to use', default='xlm-roberta-large')
    p.add_argument('--model', type=str, help='Model path.', default='save_xlm-roberta-large/lightning_logs/version_01')
    p.add_argument('--model_name', type=str, help='Model name.', default='save_xlm-roberta-large')
    p.add_argument('--stage', type=str, help='Training stage', default='fit')
    p.add_argument('--prefix', type=str, help='Prefix for storing evaluation ie final output files.', default='fit')
    p.add_argument('--batch_size', type=int, help='Batch size.', default=16)
    p.add_argument('--gpus', type=int, help='Number of GPUs.', default=1)
    p.add_argument('--cuda', type=str, help='Cuda Device', default='cuda:0')
    p.add_argument('--epochs', type=int, help='Number of epochs for training.', default=20)
    p.add_argument('--lr', type=float, help='Learning rate', default=1e-5)
    p.add_argument('--dropout', type=float, help='Dropout rate', default=0.2)
    return p.parse_args()



def get_tagset(path_to_train, path_to_dev):
    if not os.path.isfile('all_tags'):
        print('Creating all_tags file')
        label_groups = set()
        for data_file in [path_to_train, path_to_dev]:
            for fields, metadata in get_ner_reader(data=data_file):
                label_groups = label_groups.union(set(fields[-1]))

        with open('all_tags','w',encoding = 'utf-8') as f:
            f.write(str(label_groups))
    with open('all_tags') as f:
        all_tags = sorted(ast.literal_eval(f.readlines()[0]))
        wnut_iob = dict(zip((all_tags), range(len(all_tags))))
    return wnut_iob
    
    

def get_reader(file_path, max_instances=-1, max_length=50, target_vocab=None, encoder_model='xlm-roberta-large'):
    reader = CoNLLReader(max_instances=max_instances, max_length=max_length, target_vocab=target_vocab, encoder_model=encoder_model)
    reader.read_data(file_path)
    return reader



def create_model(train_data, dev_data, tag_to_id, batch_size=64, dropout_rate=0.1, stage='fit', lr=1e-5, 
                                                            encoder_model='xlm-roberta-large', num_gpus=1):
    return NERBaseAnnotator(train_data=train_data, dev_data=dev_data, tag_to_id=tag_to_id, batch_size=batch_size, stage=stage, 
                encoder_model=encoder_model, dropout_rate=dropout_rate, lr=lr, pad_token_id=train_data.pad_token_id,
                num_gpus=num_gpus, coarse_grained_tags_dict = coarse_grained_tags_dict)



def train_model(model, out_dir='', epochs=10, gpus=1):
    trainer = get_trainer(gpus=gpus, out_dir=out_dir, epochs=epochs)
    trainer.fit(model)
    return trainer



def get_trainer(gpus=1, is_test=False, out_dir=None, epochs=10):
    seed_everything(42)
    if is_test:
        return pl.Trainer(accelerator="gpu", devices=gpus) if torch.cuda.is_available() else pl.Trainer(val_check_interval=100)
    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator="gpu", devices=gpus, max_epochs=epochs,
                             default_root_dir=out_dir, strategy='ddp_find_unused_parameters_true')
        trainer.callbacks.append(get_lr_logger())
    else:
        trainer = pl.Trainer(max_epochs=epochs, default_root_dir=out_dir)
    return trainer



def get_model_earlystopping_callback():
    es_clb = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, verbose=True, mode='min')
    return es_clb    
def get_lr_logger():
    lr_monitor = LearningRateMonitor(logging_interval='step')
    return lr_monitor



def list_files(in_dir):
    files = []
    for r, d, f in os.walk(in_dir):
        for file in f:
            files.append(os.path.join(r, file))
    return files

def get_models_for_evaluation(path):
    if 'checkpoints' not in path:
        path = path + '/checkpoints/'
    model_files = list_files(path)
    models = [f for f in model_files if f.endswith('final.ckpt')]
    return models[0] if len(models) != 0 else None

def load_model(model_file, tag_to_id=None, stage='test'):
    if ~os.path.isfile(model_file):
        model_file = get_models_for_evaluation(model_file)
    hparams_file = model_file[:model_file.rindex('checkpoints/')] + '/hparams.yaml'
    model = NERBaseAnnotator.load_from_checkpoint(model_file, hparams_file=hparams_file, 
                                stage=stage, tag_to_id=tag_to_id, coarse_grained_tags_dict = coarse_grained_tags_dict)
    model.stage = stage
    return model, model_file



def get_out_filename(out_dir, model, prefix):
    model_name = os.path.basename(model)
    model_name = model_name[:model_name.rfind('.')]
    return '{}/{}_base_{}.tsv'.format(out_dir, prefix, model_name)

def save_model(trainer, out_dir, model_name='', timestamp=None):
    out_dir = out_dir + '/lightning_logs/version_' + str(trainer.logger.version) + '/checkpoints/'
    if timestamp is None:
        timestamp = time.time()
    os.makedirs(out_dir, exist_ok=True)
    outfile = out_dir + '/' + model_name + '_timestamp_' + str(timestamp) + '_final.ckpt'
    trainer.save_checkpoint(outfile, weights_only=True)
    logger.info('Stored model {}.'.format(outfile))
    return outfile



def write_eval_performance(eval_performance, out_file):
    outstr = ''
    added_keys = set()
    for out_ in eval_performance:
        for k in out_:
            if k in added_keys or k in ['results', 'predictions']:
                continue
            outstr = outstr + '{}\t{}\n'.format(k, out_[k])
            added_keys.add(k)
    
    open(out_file, 'wt').write(outstr)
    logger.info('Finished writing evaluation performance for {}'.format(out_file))
