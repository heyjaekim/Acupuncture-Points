from Image_Process_utils import *
import torchvision
import torch
import json, os
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from model_utils import *
import time, datetime
import shutil
from tb_utils import * 

# train modules 


def run():
    pass

def train(model, train_loader, optimizer, coord_loss, class_loss, epoch, writer, micro_step, t_step, lambda1=1, lambda2 = 1, enable_gpu = True, 
         *args, **kwargs):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_cls = AverageMeter()
    losses_x = AverageMeter()
    losses_y = AverageMeter()
    total_loss = AverageMeter()
    
    # train mode
    model.train()

    end = time.time()
    for batch_idx, sample in enumerate(train_loader):
        # measure loading time
        data_time.update(time.time() - end)

        # target
        cls_target = sample['target'][:,0].to(torch.float32).to('cuda')
        x_target = sample['target'][:,1].to(torch.float32).to('cuda')
        y_target = sample['target'][:,2].to(torch.float32).to('cuda')

        imgs = sample['image'].to('cuda')

        # clear all gradients
        optimizer.zero_grad()

        # compute output
        out1, out2 = model(imgs)

        # loss 
        loss_cls = class_loss( out1[:,0], cls_target )
        loss_x =  coord_loss( out2[:,0], x_target )
        loss_y = coord_loss (out2[:,1], y_target )

        loss = lambda1 * loss_cls + lambda2 * (loss_x + loss_y)

        # backpropgation
        loss.backward() 

        # update parameters
        optimizer.step()     

        # update monitors 
        losses_cls.update(  loss_cls.item(), imgs.size(0) )
        losses_x.update(  loss_x.item(), imgs.size(0) )
        losses_y.update(  loss_y.item(), imgs.size(0) )
        total_loss.update(  loss.item(), imgs.size(0) )
                 
        # measure elapsed time 
        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx % 10 == 0) & (batch_idx !=0):
            print(f'Epoch[{epoch+1}] Batch {batch_idx}/{len(train_loader)} '
                f' Time {batch_time.avg:.3f}s'
                f' Data {data_time.avg:.3f}s'
                    f' || LOSS: CLS {losses_cls.avg:.3f} X { losses_x.avg:.2f} Y { losses_y.avg:.2f} tot {total_loss.avg:.3f}'
                    f' | Current: {str(datetime.datetime.now().time())[:8]}')
            
            write_scalar_val( losses_cls.avg, writer,'TrainLossMicro/1.cls', micro_step)
            write_scalar_val( losses_x.avg, writer,'TrainLossMicro/2.X', micro_step )
            write_scalar_val( losses_y.avg, writer,'TrainLossMicro/3.Y', micro_step)
            write_scalar_val( total_loss.avg, writer,'TrainLossMicro/4.Total', micro_step )
            micro_step += 1

    write_scalar_val( losses_cls.avg, writer,'TrainLoss/1.cls', t_step)
    write_scalar_val( losses_x.avg, writer,'TrainLoss/2.X', t_step )
    write_scalar_val( losses_y.avg, writer,'TrainLoss/3.Y', t_step)
    write_scalar_val( total_loss.avg, writer,'TrainLoss/4.Total', t_step )
    t_step += 1 

    return (losses_cls.avg, losses_x.avg, losses_y.avg, total_loss.avg), micro_step, t_step

def validate(model, valid_loader, coord_loss, class_loss, epoch, writer, v_micro, v_step, lambda1=1, lambda2 = 1):
    batch_time = AverageMeter()
    losses_cls = AverageMeter()
    losses_x = AverageMeter()
    losses_y = AverageMeter()
    total_loss = AverageMeter()

    # evaluation mode 
    model.eval()

    end = time.time()
    #g_step = 0

    for batch_idx, sample in enumerate(valid_loader):
        # target
        cls_target = sample['target'][:,0].to(torch.float32).to('cuda')
        x_target = sample['target'][:,1].to(torch.float32).to('cuda')
        y_target = sample['target'][:,2].to(torch.float32).to('cuda')

        #imgs
        imgs = sample['image'].to('cuda')

        # compute output
        out1, out2 = model(imgs)

        # loss 
        loss_cls = class_loss( out1[:,0], cls_target )
        loss_x =  coord_loss( out2[:,0], x_target )
        loss_y = coord_loss (out2[:,1], y_target )

        loss = lambda1 * loss_cls + lambda2 * (loss_x + loss_y)

        # update monitors 
        losses_cls.update(  loss_cls.item(), imgs.size(0) )
        losses_x.update(  loss_x.item(), imgs.size(0) )
        losses_y.update(  loss_y.item(), imgs.size(0) )
        total_loss.update(  loss.item(), imgs.size(0) )

        # measure elapsed time 
        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx % 10 == 0) & (batch_idx !=0):
            print(f'Epoch[{epoch+1}] Batch {batch_idx}/{len(valid_loader)} '
                f' Time {batch_time.avg:.3f}s'
                    f' || LOSS: CLS {losses_cls.avg:.3f} X { losses_x.avg:.2f} Y { losses_y.avg:.2f} tot {total_loss.avg:.3f}'
                    f' | Current: {str(datetime.datetime.now().time())[:8]}')
        
        if (batch_idx % 4 == 0) & (batch_idx !=0):
            img_list = get_img_list(sample, out2)
     #       print('Writing images..')
            write_img_grid(img_list, writer, 'valid_image', v_micro)
            v_micro+=1

    write_scalar_val( losses_cls.avg, writer,'ValidLoss/1.cls', v_step)
    write_scalar_val( losses_x.avg, writer,'ValidLoss/2.X', v_step )
    write_scalar_val( losses_y.avg, writer,'ValidLoss/3.Y', v_step)
    write_scalar_val( total_loss.avg, writer,'ValidLoss/4.Total', v_step )
    v_step += 1

    return (losses_cls.avg, losses_x.avg, losses_y.avg, total_loss.avg), v_micro, v_step

def test(org_list, target_list, model, writer, name, step):
    '''
    input: tensor-valued image lists
    '''
    imlist = []
    for i in range(len(org_list)):
        _, out2 = model(org_list[i].unsqueeze(0).to('cuda'))
        target = tensor_to_np(out2)[0]
        img = tensor_to_img(target_list[i])
        img = create_circle_patch(img, target[0], target[1], color = 'red')
        fig = plt.figure(figsize = (4,4))
        plt.imshow(img)
        colors = ['blue', 'red']
        labels = ['GT', 'Predicted']
        dots = [Line2D([0], [0], marker = 'o', color = c, linestyle = 'None', markersize = 5) for c in colors]
        plt.legend(dots, labels, prop = {'size':9})
        fig.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = PIL.Image.open(buf)
        #image = ToTensor()(image) # 
        plt.close('all')
        image = ToTensor()(image)
        imlist.append(image)

    print('writing test img with step', step)
    write_img_grid(imlist, writer, name, step)
    step += 1
    return step 


def save_checkpoint(state, is_best, prefix):
    filename='./checkpoints/%s_checkpoint.pth.tar'%prefix
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/%s_model_best.pth.tar'%prefix)


# loss criterion

def coord_loss(kw = 'mse'):
    if kw == 'mse':
        return nn.MSELoss()
    if kw == 'smoothl1':
        return nn.SmoothL1Loss()

def class_loss():
    return nn.BCEWithLogitsLoss()



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count