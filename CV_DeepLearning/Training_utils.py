# DataLoader
from Image_Process_utils import *

class CoordDataSet (Dataset):
    def __init__(self, json_file, rootdir, train = True, transform = None):
        ''' current directory must include a json-file
        '''
        #super(CoordDataSet, self).__init__(rootdir, transform = transform)
        with open(json_file) as f:
            self.json_data = json.load(f)
        self.img_dir = rootdir
        self.transform = transform

    def __len__(self):
        return len(self.json_data['image'])
    
    def __getitem__(self, idx):
        img_name = self.json_data['image'][idx]['im_id']
        img_path = os.path.join(self.img_dir,img_name)
        img = Image.open(img_path + '.jpg')
        
        # output
        y_class = torch.tensor( self.json_data['image'][idx]['class'] ) 
        y_x = torch.tensor( self.json_data['image'][idx]['coord'][0] )
        y_y = torch.tensor( self.json_data['image'][idx]['coord'][1]  )
        target = [y_class, y_x, y_y ]
        
        if self.transform:
            img = self.transform(img)
            
        return img, target 


