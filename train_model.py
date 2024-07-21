##baseline from semeval 2023 conll task: https://github.com/amzn/multiconer-baseline/blob/main/train_model.py
import time, os

from utils.utils import get_reader, train_model, create_model, save_model, parse_args, get_tagset
from pytorch_lightning.utilities.model_summary import ModelSummary

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__': 
    timestamp = time.time()
    config_params = parse_args()
    out_dir_path = config_params.out_dir + '/' + config_params.model_name

    train_data = get_reader(file_path=config_params.train, target_vocab=get_tagset(config_params.train, config_params.dev), 
                    encoder_model=config_params.encoder_model, max_instances=config_params.max_instances, 
                    max_length=config_params.max_length)
    dev_data = get_reader(file_path=config_params.dev, target_vocab=get_tagset(config_params.train, config_params.dev), 
                    encoder_model=config_params.encoder_model, max_instances=config_params.max_instances, 
                    max_length=config_params.max_length)

    model = create_model(train_data=train_data, dev_data=dev_data, tag_to_id=train_data.get_target_vocab(),
                    dropout_rate=config_params.dropout, batch_size=config_params.batch_size, 
                    stage=config_params.stage, lr=config_params.lr,
                    encoder_model=config_params.encoder_model, num_gpus=config_params.gpus)

    trainer = train_model(model=model, out_dir=out_dir_path, epochs=config_params.epochs)
    
    out_model_path = save_model(trainer=trainer, out_dir=out_dir_path, model_name=config_params.model_name, timestamp=timestamp)