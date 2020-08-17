import io
import matplotlib.pyplot as plt
import torchvision
import PIL.Image
from torchvision.transforms import ToTensor
from Image_Process_utils import *  
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter


def monitor_change(img, target, predicted, label):
    '''
    img : numpy image
    '''
    #img_list = []
    # process images
    fig = plt.figure(figsize = (5,5))
    fig.suptitle(label, y=1)
    img = create_circle_patch(img, target[0], target[1], color = 'blue') # x,y 
    img = create_circle_patch(img, predicted[0], predicted[1], color= 'red') # x,y hat
    plt.imshow(img)
    # dots 
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
    return image


def get_img_list(sample, model_output, max_grid = 4):
    '''
    inputs tensor sample and model outputs (tensor)
    returns list of image tensor 
    '''
    img_list = []
    for i in range(max_grid):
        img = tensor_to_img(sample['image'][i])
        target = tensor_to_np(sample['target'][i][1:])
        predicted = tensor_to_np(model_output[i])
        label = sample['label'][i]

        img_proccessed = monitor_change(img, target, predicted, label)
        img_proccessed = ToTensor()(img_proccessed)
        img_list.append(img_proccessed)
    return img_list


def tensor_to_img(img_tensor):
    return np.transpose( to_numpy(img_tensor), (1,2,0))

def tensor_to_np(tensor):
    return to_numpy(tensor)


def write_img_grid(img_list, writer, name, step):
    imgs_grid = torchvision.utils.make_grid(img_list, normalize = False)
    writer.add_image(name, imgs_grid, global_step = step)
   # writer.flush()

def write_scalar_val(scalar, writer, name, step):
    writer.add_scalar(name, scalar, global_step = step)
    # writer.flush()