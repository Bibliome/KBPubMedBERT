from argparse import ArgumentParser


def get_args():
     parser = ArgumentParser(description='Finetuning BERT models')
	
     group = parser.add_argument_group('--bert_options')
     group.add_argument("--data_path", default=None, type=str,required=True,
                         help="The input data directory. Should contain data files (.pkl) for the task; in case of "
                              "using graph embeddings, should contain embedding files (.npy).")
     group.add_argument("--input_filename",type=str)
     group.add_argument("--mode",type=str,required=True,
                         help="no_kb (only BERT) or with_kb (with knowledge base).")
     group.add_argument("--inference_only",action="store_true",
                         help="only inference. If set, you need to specify the path to a pre-trained model checkpoint.")
     group.add_argument("--checkpoint_path",default=None,type=str,
                         help="set this only when inference_only is set to true. If inference_only is set but no checkpoint path "
                              "is given, the checkpoint path will by default be set to corpus_mode_bertName_learningRate_seed")
     group.add_argument("--model_type",default="pubmedbert",type=str,
                         help="abbreviation of model to use")
     group.add_argument("--model_path",default="./models/",type=str)
     group.add_argument("--logging_path",default="./logging/",type=str)
     group.add_argument("--pretrained_model_path", default="./pretrained_models/", type=str,
                         help="Path to pre-trained model or shortcut name")
     group.add_argument("--force_cpu",action="store_true",
                         help="if set, the script will be run WITHOUT GPU.")
     group.add_argument("--config_path", default="./config/", type=str,
                         help="Path to pre-trained config or shortcut name selected in the list")
     group.add_argument("--corpus_name", default=None, type=str)
     group.add_argument("--tmp_path",default="./tmp/",type=str)
     group.add_argument("--output_path", default="./predictions/", type=str,
                         help="The output directory where the model predictions and checkpoints will be written.")
     group.add_argument("--num_labels",type=int,default=2,
                         help="number of layers in the last linear layer")
     group.add_argument("--dry_run",action="store_true")
     group.add_argument("--number_of_examples_for_dry_run",type=int,default=50)
     group.add_argument("--monitor",type=str,default="score",
                         help="criteria to use for early stopping")
     group.add_argument("--early_stopping",action="store_true",
                         help="if use early stopping during training")
     group.add_argument("--patience",type=int,default=3,
                         help="patience of early stopping")
     group.add_argument("--seq_length", default=128, type=int,
                         help="The maximum total input sequence length after tokenization. Sequences longer "
                              "than this will be truncated, sequences shorter will be padded.")
     group.add_argument("--max_length",type=int,default=128)
     group.add_argument("--batch_size", default=16, type=int,
                         help="Batch size per GPU/CPU for training.")
     group.add_argument("--learning_rate",type=float,default=2e-5)
     group.add_argument("--extra_learning_rate",type=float,default=2e-5,
                         help="learning rate for added linear layers")
     group.add_argument("--max_grad_norm", default=1.0, type=float,
                         help="Max gradient norm.")
     group.add_argument("--max_num_epochs",default=15,type=int,
                         help="maximum number of epochs")
     group.add_argument("--num_train_epochs", default=20, type=int,
                         help="Total number of training epochs to perform.")

     group.add_argument("--seed",type=int,default=42)
     group.add_argument("--shuffle_train",action="store_true",help="if set, shuffle the train set before training.")
     group.add_argument("--warmup",action="store_true",help="if set, use linear warmup scheduler for learning rate.")
     group.add_argument("--warmup_ratio",type=float,default=0.1)
     group.add_argument('--logging_steps', type=int, default=50,
                         help="Log every X updates steps.")
     #group.add_argument("--debug",action="store_true")
     args = parser.parse_args()
     return args
