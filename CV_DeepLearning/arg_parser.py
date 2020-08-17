def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def common_arg_parser():
    parser = arg_parser()

    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('--prefix', default = 'Hi', type = str, help = 'prefix for logging & ckpt save')
    parser.add_argument('--model', default='resnet34', type=str, help='Model Architecture (default: resnet34)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='starting epoch')
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    return parser