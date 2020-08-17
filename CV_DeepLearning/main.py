import os
from Image_Process_utils import *
from model_utils import *
from Dataset_utils import *
from Training_utils import *
from arg_parser import common_arg_parser
from tb_utils import *

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

best_val = 100000000

def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()
    model = create_model(args.model)
    #summary(model.to('cuda'), input_size = (3, 700, 700))

    # Load dataset 
    img_dir = './Acu_Dataset/sotack/org'
    json_file = './Acu_Dataset/sotack/sotack_info.json'

    my_transforms = transforms.Compose([
    #transforms.Normalize((0.5,), (0.5,)),
    Rescale(256),
    Tensorize()
    ])
    sotack_dataset = HandDataSet(json_file, img_dir, transform = my_transforms, train=True) # pytorch 로 변환

    # split dataset



    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_val = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    

    pass

if __name__ == '__main__':
    main(sys.argv)