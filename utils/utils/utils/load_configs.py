import argparse
import sys
import torch
import os


def get_link_prediction_args(is_evaluation: bool = False):
    """
    Get the args for the link prediction task.
    """
    parser = argparse.ArgumentParser(description="Interface for the link prediction task")
    
    # Ensure parameters are only defined once
    existing_args = [action.dest for action in parser._actions]
    if 'channel_embedding_dim' not in existing_args:
        parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    if 'num_heads' not in existing_args:
        parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
     # Add task_name parameter
    parser.add_argument('--task_name', type=str, default='link_prediction', choices=['link_prediction', 'node_classification'],
                        help='task to perform, either link_prediction or node_classification')
    # Other parameters...
    parser.add_argument('--model_name', type=str, choices=['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'EdgeBank', 'TCL', 'GraphMixer', 'DyGFormer', 'DIFFDyG','PathDyG', 'NPTDyG','DeformableDyG','CARDDyG', 'TIMERDyG', 'STDyGFormer', 'STDIFFDyG'],
                        help='name of the model, note that EdgeBank is only applicable for evaluation')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='wikipedia',
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci', 'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id to use, e.g. 0')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--edge_bank_memory_mode', type=str, default='unlimited_memory', help='how memory of EdgeBank works',
                        choices=['unlimited_memory', 'time_window_memory', 'repeat_threshold_memory'])
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion', help='how to select the time window size for time window memory',
                        choices=['fixed_proportion', 'repeat_interval'])
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--max_input_sequence_length', type=int, default=512, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=3, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')
   

    try:
        args = parser.parse_args()
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        parser.print_help()
        sys.exit()
    if args.task_name == 'link_prediction':
       print("Running link prediction task...")
    # 执行链路预测任务的逻辑
    elif args.task_name == 'node_classification':
        print("Running node classification task...")
    # 执行节点分类任务的逻辑
    if args.model_name == 'EdgeBank':
        assert is_evaluation, 'EdgeBank is only applicable for evaluation!'

    if args.load_best_configs:
        load_link_prediction_best_configs(args=args)

    return args


def load_link_prediction_best_configs(args):
    """
    Load the best configurations for the link prediction task.
    """
    if args.model_name == 'TGAT':
        args.num_heads = 2
        args.num_layers = 2
        args.time_feat_dim = 16
        args.dropout = 0.1
        args.learning_rate = 0.0001
        args.weight_decay = 1e-5
        args.patience = 5
    elif args.model_name == 'DyGFormer':
        args.num_heads = 4
        args.num_layers = 3
        args.channel_embedding_dim = args.channel_embedding_dim or 32  # 保留命令行参数
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0005
        args.weight_decay = 1e-4
        args.patience = 10
    elif args.model_name == 'STDyGFormer':
        args.num_heads = 4
        args.num_layers = 3
        args.channel_embedding_dim = args.channel_embedding_dim or 32  # 保留命令行参数
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0005
        args.weight_decay = 1e-4
        args.patience = 10
    elif args.model_name == 'DIFFDyG':
        args.num_heads = 12  # 保留命令行参数
        args.num_layers = 3
        args.channel_embedding_dim =  32  
        args.time_feat_dim = 16
        args.dropout = 0.2
        args.learning_rate = 0.0005
        args.weight_decay = 5e-3  # 进一步增大weight_decay来提升inductive性能
        args.patience = 10
    elif args.model_name == 'STDIFFDyG':
        args.num_heads = 12  # 保留命令行参数
        args.num_layers = 3
        args.channel_embedding_dim =  32  
        args.time_feat_dim = 16
        args.dropout = 0.2
        args.learning_rate = 0.0005
        args.weight_decay = 1e-3  # 增大weight_decay来提升inductive性能
        args.patience = 10
    elif args.model_name == 'PathDyG':
        args.num_heads = 4
        args.num_layers = 2
        args.channel_embedding_dim = 64
        args.time_feat_dim = 16
        args.dropout = 0.3
        args.learning_rate = 0.0005
        args.weight_decay = 1e-4
        args.patience = 10
    elif args.model_name == 'NPTDyG':
        args.num_heads = 4
        args.num_layers = 2
        args.time_feat_dim = 16
        args.channel_embedding_dim = 48
        args.dropout = 0.2
        args.learning_rate = 0.0001
        args.weight_decay = 1e-5
        args.patience = 10
        # 确保 max_input_sequence_length 是 num_heads 的整数倍
        args.max_input_sequence_length = 512  # 512 是 8 的整数倍
    elif args.model_name == 'DeformableDyG':
        args.num_neighbors = 20
        args.num_layers = 2
        args.num_heads = 2
        args.channel_embedding_dim = 64
        args.time_feat_dim = 100
        args.dropout = 0.2
        
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
        else:
            args.max_input_sequence_length = 32
        
        if args.dataset_name in ['reddit']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
    elif args.model_name == 'CARDDyG':
        args.time_feat_dim = 100  # 确保参数已设置
        args.channel_embedding_dim = 64
        args.d_model = args.channel_embedding_dim  # 添加 d_model 参数
        args.num_heads = 8
        args.num_layers = 4
        args.e_layers = 2  # 添加 e_layers 参数
        args.dropout = 0.1
        args.max_input_sequence_length = 512
        args.patch_len = 16  # 添加 patch_len 参数
        args.stride = 1  # 添加 stride 参数
    elif args.model_name == 'TIMERDyG':
        args.num_heads = 4
        args.num_layers = 3
        args.channel_embedding_dim = args.channel_embedding_dim or 32
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0005
        args.weight_decay = 1e-4
        args.patience = 10
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")


def get_node_classification_args():
    """
    get the args for the node classification task
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the node classification task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='wikipedia', choices=['wikipedia', 'reddit'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--model_name', type=str, default='DyGFormer', help='name of the model',
                        choices=['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer','DIFFDyG', 'PathDyG','NPTDyG','DeformableDyG','CARDDyG', 'TIMERDyG', 'STDyGFormer', 'STDIFFDyG'])
    parser.add_argument('--gpu', type=int, default=0, help='gpu id to use, e.g. 0')
    parser.add_argument('--pretrained_model_name', type=str, help='Name of the pretrained model to load (e.g., DIFFDyG_seed1)')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=32, help='maximal length of the input sequence of each node')
    # NPTDyG特定参数
    parser.add_argument('--mask_prob', type=float, default=0.15, help='NPTDyG: 遮蔽概率')
    parser.add_argument('--augmentation_mask_prob', type=float, default=0.15, help='NPTDyG: 数据增强时的遮蔽概率')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=240, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')

    try:
        args = parser.parse_args()
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    assert args.dataset_name in ['wikipedia', 'reddit'], f'Wrong value for dataset_name {args.dataset_name}!'
    if args.load_best_configs:
        load_node_classification_best_configs(args=args)

    return args


def load_node_classification_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the node classification task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name == 'TGAT':
        args.num_neighbors = 20
        args.num_layers = 2
        args.dropout = 0.1
        if args.dataset_name in ['reddit']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
        args.num_neighbors = 10
        args.num_layers = 1
        args.dropout = 0.1
        args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'CAWN':
        args.time_scaling_factor = 1e-6
        args.num_neighbors = 32
        args.dropout = 0.1
        args.sample_neighbor_strategy = 'time_interval_aware'
    elif args.model_name == 'TCL':
        args.num_neighbors = 20
        args.num_layers = 2
        args.dropout = 0.1
        if args.dataset_name in ['reddit']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'GraphMixer':
        args.num_layers = 2
        if args.dataset_name in ['reddit']:
            args.num_neighbors = 10
        else:
            args.num_neighbors = 30
        args.dropout = 0.5
        args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'DyGFormer':
        args.num_layers = 3  # 从错误信息看，预训练模型有3层
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        else:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ['reddit']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
        
        # 为了兼容预训练模型，更新特征维度参数（这些值从错误信息中提取）
        if args.dataset_name == 'wikipedia':
            args.node_feat_dim = 688
            args.time_feat_dim = 400
            args.neighbor_co_occurrence_dim = 200
        # 其他数据集保持默认值
    elif args.model_name == 'STDyGFormer':
        args.num_layers = 3  # 从错误信息看，预训练模型有3层
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        else:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ['reddit']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
        
        # 为了兼容预训练模型，更新特征维度参数（这些值从错误信息中提取）
        if args.dataset_name == 'wikipedia':
            args.node_feat_dim = 688
            args.time_feat_dim = 400
            args.neighbor_co_occurrence_dim = 200
        # 其他数据集保持默认值
    elif args.model_name == 'DIFFDyG':
        args.num_layers = 3
        args.num_heads = 12 
        args.channel_embedding_dim = 50
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        else:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ['reddit']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
        # 添加weight_decay以提升inductive性能
        args.weight_decay = 1e-3  # 增大weight_decay来提升泛化能力
        args.learning_rate = 0.0001  # 适当降低学习率配合更大的weight_decay
        if args.dataset_name == 'wikipedia':
            args.node_feat_dim = 688
            args.time_feat_dim = 400
            args.neighbor_co_occurrence_dim = 200
    elif args.model_name == 'STDIFFDyG':
        args.num_layers = 3
        args.num_heads = 12 
        args.channel_embedding_dim = 50
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        else:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ['reddit']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
        # 添加weight_decay以提升inductive性能
        args.weight_decay = 1e-3  # 增大weight_decay来提升泛化能力
        args.learning_rate = 0.0001  # 适当降低学习率配合更大的weight_decay
        if args.dataset_name == 'wikipedia':
            args.node_feat_dim = 688
            args.time_feat_dim = 400
            args.neighbor_co_occurrence_dim = 200
    elif args.model_name == 'PathDyG':
        args.num_layers = 3
        # 为PathDyG模型使用固定配置，避免维度不匹配问题
        args.channel_embedding_dim = 32  # 预训练模型使用32维
        args.time_feat_dim = 16  # 预训练模型使用16维时间特征
        args.max_input_sequence_length = 1024  # 预训练模型使用1024，需要匹配W_pos维度
        
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 1024
            args.patch_size = 2
        else:
            args.max_input_sequence_length = 1024  # 使用1024匹配预训练模型
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ['reddit']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
    elif args.model_name == 'NPTDyG':
        args.num_layers = 2
        args.num_heads = 4
        args.channel_embedding_dim = 48
        args.time_feat_dim = 16
        # NPTDyG特定参数
        args.mask_prob = 0.15
        args.augmentation_mask_prob = 0.15
        
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        else:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        
        if args.dataset_name in ['reddit']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
    elif args.model_name == 'DeformableDyG':
        args.num_neighbors = 20
        args.num_layers = 2
        args.num_heads = 2
        args.channel_embedding_dim = 32
        args.time_feat_dim = 50
        args.dropout = 0.2
        
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
        else:
            args.max_input_sequence_length = 32
        
        if args.dataset_name in ['reddit']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
    elif args.model_name == 'TIMERDyG':
        args.num_heads = 4
        args.num_layers = 3
        args.channel_embedding_dim = args.channel_embedding_dim or 32
        args.patch_size = 4
        args.dropout = 0.2
        args.learning_rate = 0.0005
        args.weight_decay = 1e-4
        args.patience = 10
    elif args.model_name == 'CARDDyG':
        # 你可以根据实际经验或参考 DyGFormer/PathDyG 的配置来设置
        args.channel_embedding_dim = 64
        args.patch_size = 4
        args.num_layers = 2
        args.num_heads = 8
        args.dropout = 0.1
        args.max_input_sequence_length = 20
        # 其它你希望设置的参数
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")
