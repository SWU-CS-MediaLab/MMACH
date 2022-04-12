from PIL import Image
from torchvision import transforms
import os

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
my_transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])
class ReadImage:
    def __init__(self,dir):
        super(ReadImage, self).__init__()
        self.transform = my_transform
        self.img_dir = dir
    def read_img(self,img_name):
        image_url = os.path.join(self.img_dir, img_name)
        image = Image.open(image_url).convert('RGB')
        image = self.transform(image)
        return image
