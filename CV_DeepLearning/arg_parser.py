def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def common_arg_parser():
    parser = arg_parser()

    parser.add_argument('--kw', default= 'hapgok')
    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('--prefix', default = '', type = str, help = 'prefix for logging & ckpt save')
    parser.add_argument('--model', default='resnet34', type=str, help='Model Architecture (default: resnet34)')
    parser.add_argument('--epochs', default=80, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='starting epoch')
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--test_ratio', type = float, default = '0.01')
    parser.add_argument('--valid_ratio', type = float, default = '0.1')  

    parser.add_argument('--bs', type = int, default= '32')
    parser.add_argument('--lr', type = float, default= '0.001')
    parser.add_argument('--loss_type', default = 'mse')
    parser.add_argument('--rescale', type = int, default = '256')

    parser.add_argument('--decay', type = float, default = '0.9')
    parser.add_argument('--decay_step', type = int, default= 1)
    parser.add_argument('--betas', type = float, default= (0.5, 0.999) )
    return parser