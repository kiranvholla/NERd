1. We use the baseline version of multiconer as our starting code: https://github.com/amzn/multiconer-baseline. On top of it we build in over 10+ improving feature-sets.

2. train_model.py can be used for training; After that predict_tags.py can be used for predictions or evaluate.py can be used for evaluations

3. The arguments are unchanged from the baseline. They can be found in utils.py file in utils folder. The 2 main arguements are: 
- "stage" needs to be "fit" or "test"
- If "test" stage, then "model" needs to point to the path of the checkpoint: For e.g. save_xlm-roberta-large/lightning_logs/version_2/. The code searches for "checkpoints" folder in this dir and from there picks the checkpoint having the "final" suffix. 
- There is no need to set stage = predict. For predictions, retain stage as "test" and simply run predict_tags.py while ensuring that "model" argument has the right value.

The libraries needed are:
- python 3.8.12
- numpy: 1.22.2
- pandas: 1.5.2
- overrides (7.3.1)

Also, Please install in following order as allennlp overides some of the pytorch files:
- torch (1.12.1)
- allennlp (version 2.10.1)
- pytorch-lightning (2.0.1)


Additional Notes:
=====================================================================================
Note 1: If executing on Windows, please set num_workers in dataloaders to 0. This is because of lack of support for spawning & multi-processing in Windows + certain (newer) Python versions

Note 2: With DEBERTA if below error is generated:
- AttributeError: module 'numpy' has no attribute 'int'. `np.int` was a deprecated alias for the builtin `int`. To avoid this error in existing code, use `int` by itself. 
- HF has not yet fixed this problem in their transformers. This happens on latest numpy versions (1.24 and above). Downgrade numpy to anything between 1.20 and 1.24 so for e.g. pip install --upgrade numpy==1.23.5 works fine
