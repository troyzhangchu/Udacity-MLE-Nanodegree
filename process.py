###########
# @TroyZhang
# 6/30/2021
###########

import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import numpy as np

class ProcessImage(object):
    def __init__(self):
        # load model
        self.model = None
        self.load_model()

        # human detector
        self.face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

        # classes
        self.classes = []
        with open('classes.txt', 'r') as f:
            self.classes = f.read()
        self.classes = [x.strip() for x in self.classes.split('/n')]

    def load_model(self):
        self.model = models.resnet18(pretrained=False)
        num_fc_in = self.model.fc.in_features
        self.model.fc = nn.Linear(num_fc_in, 133)
        self.model.load_state_dict(torch.load('model_transfer.pt', map_location=torch.device('cpu')))

    # process image
    def process_image(self, img_path):
        # Load Image
        img = Image.open(img_path)
        # Get the dimensions of the image
        width, height = img.size   
        # Resize by keeping the aspect ratio, but changing the dimension
        # so the shortest size is 255px
        img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))   
        # Get the dimensions of the new image size
        width, height = img.size   
        # Set the coordinates to do a center crop of 200 x 200
        left = (width - 200)/2
        top = (height - 200)/2
        right = (width + 200)/2
        bottom = (height + 200)/2
        img = img.crop((left, top, right, bottom))   
        # Turn image into numpy array
        img = np.array(img)   
        # Make the color channel dimension first instead of last (PID has a different order of the colors)
        img = img.transpose((2, 0, 1))   
        # Make all values between 0 and 1
        img = img/255   
        # Normalize based on the preset mean and standard deviation
        img[0] = (img[0] - 0.485)/0.229
        img[1] = (img[1] - 0.456)/0.224
        img[2] = (img[2] - 0.406)/0.225
        # Add a fourth dimension to the beginning to indicate batch size
        img = img[np.newaxis,:]
        # Turn into a torch tensor
        image = torch.from_numpy(img)
        image = image.float()
        return image

    def face_detector(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def VGG16_predict(self, img_path):
        img = self.process_image(img_path)
        output = models.resnet18(pretrained=True).forward(img)
        # output = models.vgg16(pretrained=True).forward(img)
        output = torch.exp(output)
        probs, clas = output.topk(1, dim=1)
        return clas.item()

    def dog_detector(self, img_path):
        pred = self.VGG16_predict(img_path)
        if pred >= 151 and pred <= 268:
            return True
        return False

    def predict_breed_transfer(self, img_path):
        img = self.process_image(img_path)
        output = self.model.forward(img)
        output = torch.exp(output)
        probs, clas = output.topk(1, dim=1)
        return self.classes[clas.item()]

    def run_app(self, img_path):
        ## handle cases for a human face, dog, and neither
        # Human or Dog
        if self.face_detector(img_path):
            breed = self.predict_breed_transfer(img_path)
            return "Hi human! ^_^ You look like a {}".format(breed)
        elif self.dog_detector(img_path):
            breed = self.predict_breed_transfer(img_path)
            return "Bark bark! It looks like a {}".format(breed)
        else:
            return "No human and no dog!"
