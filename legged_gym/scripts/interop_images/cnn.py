import torch
from torch import optim, nn
from torchvision import models, transforms

import tqdm
import cv2
import numpy as np



class FeatureExtractor(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor, self).__init__()
		# Extract VGG-16 Feature Layers
    self.features = list(model.features)
    self.features = nn.Sequential(*self.features)
		# Extract VGG-16 Average Pooling Layer
    self.pooling = model.avgpool
		# Convert the image into one-dimensional vector
    self.flatten = nn.Flatten()
		# Extract the first part of fully-connected layer from VGG16
    self.fc = model.classifier[0]
    self.fc2 = model.classifier[3]
    self.fc3 = model.classifier[6]
  
  def forward(self, x):
		# It will take the input 'x' until it returns the feature vector called 'out'
    out = self.features(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out)
    out = self.fc2(out) 
    out = self.fc3(out)  
    return out 

# Initialize the model
model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model)

# Change the device to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
new_model = new_model.to(device)

# Transform the image, so it becomes readable with the model
transform = transforms.Compose([
  #transforms.ToPILImage(),
  #transforms.CenterCrop(512),
  #transforms.Resize(448),  
  transforms.ToTensor(),                              
])

# Will contain the feature
features = []

# Iterate each image
# for i in tqdm(sample_submission.ImageID):
# Set the image path
path = 'cam-00.png'
# Read the file
# img = cv2.imread(path)
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# Transform the image
img = transform(img)
# print('t',img.shape)
img = torch.cat([img, img, img], dim=0)
print('re',img.shape)
softmax = torch.nn.Softmax(dim = 1)
img = softmax(img)
print(img[1])
print(img[1].sum())
transform1 = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        #transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])])
img = transform1(img)
print('n',img.shape)
a = img
print(a)
# print(torch.mean(a))
# Reshape the image. PyTorch model reads 4-dimensional tensor
# [batch_size, channels, width, height]
img = img.reshape(1, 3, 224, 224)
img = img.to(device)
# We only extract features, so we don't need gradient
with torch.no_grad():
  # Extract the feature from the image
  feature = new_model(img)
# Convert to NumPy Array, Reshape it, and save it to features variable
features.append(feature.cpu().detach().numpy().reshape(-1))

# Convert to NumPy Array
features = np.array(features)

print(feature.shape)