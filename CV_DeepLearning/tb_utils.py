import io
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor
from Image_Process_utils import *  
from matplotlib.lines import Line2D
from PIL import Image
from PIL import ImageDraw

def gen_train_monitor_plot(img_tensor, predicted, target, size =3):
    '''
    n : batch_size
    img_tensor : n x img_tensor 
    output : n x [coordinates]
    target : n x [ x, y ]
    '''
    img_list = []
    # process images
    for i in range(size): 
        plt.figure()
        img = np.transpose( to_numpy(img_tensor[i]), (1,2,0))
        img = create_circle_patch(img, to_numpy(target[1])[i], to_numpy(target[2])[i]) # x,y 
        img = create_circle_patch(img, to_numpy(predicted[i])[0], to_numpy(predicted[i])[1], color = 'red') # x,y
        plt.imshow(img)
        # dots 
        colors = ['blue', 'red']
        labels = ['GT', 'Predicted']
        dots = [Line2D([0], [0], marker = 'o', color = c, linestyle = 'None', markersize = 5) for c in colors]
        plt.legend(dots, labels, prop = {'size':9})
        #figure.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image) # 
        plt.close('all')
        img_list.append(image)
    return img_list