import Training_utils, torchvision
from torchvision import transforms
import tb_utils, torch, model_utils, sys, os
from tb_utils import *
from Training_utils import *
from Image_Process_utils import *
import model_utils
from model_utils import *
from Dataset_utils import *
import arg_parser
from arg_parser import common_arg_parser
from torchsummary import summary

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

global t_step
global v_step
global test_step
global micro_step
global v_micro

def today_timeinfo():
    today = str(datetime.datetime.now().today())
    return today[5:7] +today[8:10] + '_' + today[11:13] + today[14:16]

def print_dash():
    print('##################################################################')


def main(args):
    # Parser
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()
    args.prefix = args.kw + today_timeinfo() + args.prefix

    # Initialize  
    print('\n\nInitializing...\n')
    best_val = 100000000
    t_step = v_step = test_step = micro_step = v_micro = 0
    test_imlist, target_imlist =  gen_test_img_list()

    print(f'Checkpoint Name: {args.prefix}')
    print(f'Training keyword : {args.kw}')
    print(f'Current Path:{os.getcwd()} \n')

    print_dash()
    print('\n<Hyperparameters>')
    print('0. Model :', args.model)
    print('1.Batch_size :', args.bs)
    print('2.Learning rate: ', args.lr)
    print('3.Rescale: ', args.rescale)
    print(f'4.Schedule_Decay: Decaying {args.decay} per {args.decay_step} step')
    print(f'5. Epoch : {args.epochs}\n')
    print_dash()

    print('\nCreating Dataset...\n')
    
    # Transformation 
    my_transforms = transforms.Compose([
        #transforms.Normalize((0.5,), (0.5,)),
        Rescale(args.rescale),Tensorize()])

    # Dataset   
    concat_dataset = concat_augmented(kw = args.kw, transform = my_transforms)   
    train_set, valid_set, test_set = train_test_valid_splitter(concat_dataset, args.test_ratio, args.valid_ratio)

    # Data Loader
    train_loader = DataLoader(train_set, batch_size = args.bs, shuffle = True, num_workers=args.workers)
    valid_loader = DataLoader(valid_set, batch_size = args.bs, shuffle = True, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size = args.bs, shuffle = True, num_workers=args.workers)

    # Create Model
    model = create_model(args.model)

    # GPU Compatibility
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on: ', device) 

    if device != 'cpu':
        model.to('cuda')
        #summary(model.to('cuda'), input_size = (3, args.rescale, args.rescale))
        pass
    
    # optimizer and Scheduler 
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas = args.betas)
    scheduler = StepLR(optimizer, step_size = args.decay_step, gamma = args.decay)
    
    # loss function
    cls_loss = class_loss()
    crd_loss = coord_loss(kw = args.loss_type)

    # Tensorboard Writer
    writer = SummaryWriter('runs/'+args.prefix)
    

    # Resume from best results
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_val = checkpoint['best_val']
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # train loop
    for epoch in range(args.start_epoch, args.epochs):
        _, micro_step, t_step =  train(model, train_loader, optimizer, crd_loss, cls_loss, epoch, writer, micro_step, t_step)
        val_loss, v_micro, v_step = validate(model, valid_loader, crd_loss, cls_loss, epoch, writer, v_micro, v_step)
        test_step = test(test_imlist, target_imlist, model, writer, 'test_image', test_step)
        writer.flush()

        is_best = sum(val_loss[0:2]) < best_val
        best_val = min( sum(val_loss[0:2]), best_val )

        save_checkpoint({
        'epoch' : epoch,
        'model' : args.model, 
        'state_dict' : model.state_dict(),
        'best_val' : best_val, 
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()
        }, is_best, args.prefix)
    
        scheduler.step()

    args.resume = './checkpoints/ekmoon0830_2236all+sc+sc_filled_model_best.pth.tar'

    # summary of results
    print_dash()
    print(f'Training DONE on epoch: {epoch}')
    print(f'Best Validation Loss: {best_val}')


if __name__ == '__main__':
    main(sys.argv)