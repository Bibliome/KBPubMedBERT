from argparse import ArgumentParser


def get_args():
     parser = ArgumentParser(description='Finetuning BERT models')
	
     group = parser.add_argument_group('--bert_options')
     group.add_argument("--data_dir", default=None, type=str,
                         help="The input data dir. Should contain the .csv files (or other data files) for the task.")
     group.add_argument("--emb_dir", default=None, type=str)
     group.add_argument("--mode",required=True,type=str)
     group.add_argument("--kb_emb_dim", default=None, type=int)
     group.add_argument("--model_type",default="scibert",type=str,
                         help="abbreviation of model to use")
     group.add_argument("--model_name_or_path", default=None, type=str,
                         help="Path to pre-trained model or shortcut name")
     group.add_argument("--config_name_or_path", default="./config/pubmedbert.json", type=str,
                         help="Path to pre-trained config or shortcut name selected in the list")
     group.add_argument("--task_name", default=None, type=str)
     group.add_argument("--output_dir", default=None, type=str, required=True,
                         help="The output directory where the model predictions and checkpoints will be written.")
     group.add_argument("--num_labels",type=int,default=2,
                         help="number of layers in the last linear layer")
     group.add_argument("--train_dev_split",action="store_true",
                         help="whether randomly split train & dev data when loading data")
     group.add_argument("--get_ig",action="store_true")
     group.add_argument("--dry_run",action="store_true")
     group.add_argument("--number_of_examples_for_dry_run",type=int,default=50)
     group.add_argument("--monitor",type=str,default="score",
                         help="criteria to use for early stopping")
     group.add_argument("--run_id",type=int,default=0)
     group.add_argument("--early_stopping",action="store_true",
                         help="if use early stopping during training")
     group.add_argument("--patience",type=int,default=3,
                         help="patience of early stopping")
     group.add_argument("--config_name", default="", type=str,
                         help="Pretrained config name or path if not the same as model_name")
     group.add_argument("--seq_length", default=128, type=int,
                         help="The maximum total input sequence length after tokenization. Sequences longer "
                              "than this will be truncated, sequences shorter will be padded.")
     group.add_argument("--num_ensemble",type=int,
                         help="number of repetitive experiments to get an ensemble result")
     group.add_argument("--max_length",type=int,default=128)
     group.add_argument("--batch_size", default=16, type=int,
                         help="Batch size per GPU/CPU for training.")
     group.add_argument("--learning_rate",type=float,default=2e-5)
     group.add_argument("--extra_learning_rate",type=float,default=-1)
     group.add_argument("--max_grad_norm", default=1.0, type=float,
                         help="Max gradient norm.")
     group.add_argument("--max_num_epochs",default=15,type=int,
                         help="maximum number of epochs")
     group.add_argument("--num_train_epochs", default=20, type=int,
                         help="Total number of training epochs to perform.")

     group.add_argument("--seed",type=int)
     group.add_argument("--shuffle_train",action="store_true",help="if set, shuffle the train set before training.")
     group.add_argument("--warmup",action="store_true",help="if set, use linear warmup scheduler for learning rate.")
     group.add_argument("--warmup_ratio",type=float,default=0.1)
     group.add_argument("--adam",action="store_true",help="use Adam as optimizer; AdamW by default")
     group.add_argument("--test_trivial_kb_embedding",action="store_true",
                        help="if set, use a randomly initialized matrix as KB embedding; for ablation study.")
     group.add_argument('--logging_steps', type=int, default=50,
                         help="Log every X updates steps.")
     args = parser.parse_args()
     return args
