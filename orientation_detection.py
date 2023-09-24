import cv2
import numpy as np
from PIL import Image
from arrow_model import ArrowModel
from plus_minus_model import PlusMinusModel
import torch
import torchvision.transforms as transforms
import os
import torchvision.models as models


def sharpen_image(img):
    # Create our shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(img, -1, kernel_sharpening)
    return sharpened

def get_current_direction(image_path):

    # Read the image
    img = cv2.imread(image_path)

    img = sharpen_image(img)


    # Convert to grayscale

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)





    # find contours

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    contours = sorted(contours, key=cv2.contourArea)



        
    x, y, w, h = cv2.boundingRect(contours[0])


    arrow = img[y:y+h, x:x+w]



    cv2.imwrite('current_arrow_temp.png', arrow)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    arrow_model = ArrowModel()


    classes = ('LEFT', 'RIGHT', 'UP', 'DOWN')

    arrow_model.load_state_dict(torch.load('arrow_model.pth'))

    arrow_model.to(device)

    arrow_model.eval()

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # inference on a single image

    img = Image.open('./current_arrow_temp.png').convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)


    outputs = arrow_model(img)
    _, predicted = torch.max(outputs, 1)

    # print(f'Predicted: {classes[predicted[0]]}')
    
    
    # delete the temporary image
    os.remove('current_arrow_temp.png')
    
    return classes[predicted[0]]


def get_voltage_direction(image_path):
    # Read the image
    img = cv2.imread(image_path)

    img = sharpen_image(img)


    # Convert to grayscale

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)





    # find contours

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=cv2.contourArea)
    
    x1, y1, w, h = cv2.boundingRect(contours[0])
    
    padding = 10
    
    symbol1 = img[y1-padding:y1+h+padding, x1-padding:x1+w+padding]
    
    
    
    
    
    x2, y2, w, h = cv2.boundingRect(contours[1])
    
    symbol2 = img[y2-padding:y2+h+padding, x2-padding:x2+w+padding]
    
    
    cv2.imwrite('symbol1_temp.png', symbol1)
    
    cv2.imwrite('symbol2_temp.png', symbol2)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    
    classes = ('PLUS', 'MINUS', 'VERTICAL_MINUS')
    
    plus_minus_model = models.resnet50()
    
    fc = torch.nn.Linear(plus_minus_model.fc.in_features , 3)
    
    plus_minus_model.fc = fc
    
    plus_minus_model.load_state_dict(torch.load('plus_minus_classifier.pt'))
    
    # plus_minus_model.to(device)
    
    plus_minus_model.eval()
    
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
                           transforms.Resize(pretrained_size),
                           transforms.CenterCrop(pretrained_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means, 
                                                std = pretrained_stds)
                       ])

    
    
    
    symbol1_img = Image.open('./symbol1_temp.png').convert('RGB')
    symbol1_img = transform(symbol1_img)
    symbol1_img = symbol1_img.unsqueeze(0)
    
    
    outputs = plus_minus_model(symbol1_img)
    _, predicted = torch.max(outputs, 1)
    
    predicted_class1 = classes[predicted[0]]
    
    
    
    
    
    symbol2_img = Image.open('./symbol2_temp.png').convert('RGB')
    symbol2_img = transform(symbol2_img)
    symbol2_img = symbol2_img.unsqueeze(0)
    
    outputs = plus_minus_model(symbol2_img)
    _, predicted = torch.max(outputs, 1)
    
    predicted_class2 = classes[predicted[0]]
    
    
        
    # delete the temporary image
    os.remove('symbol1_temp.png')
    os.remove('symbol2_temp.png')
    
    if predicted_class1 == 'VERTICAL_MINUS':
        predicted_class1 = 'MINUS'
        
    if predicted_class2 == 'VERTICAL_MINUS':
        predicted_class2 = 'MINUS'
        
    
    tolerance = 30
    
    print(predicted_class1, predicted_class2)
    
    if abs(y1-y2) < tolerance and x1 >= x2 and predicted_class1 == 'PLUS' and predicted_class2 == 'MINUS':
        return 'RIGHT'
    
    if abs(y1-y2) < tolerance and x1 <= x2 and predicted_class1 == 'PLUS' and predicted_class2 == 'MINUS':
        return 'LEFT'
    
    if abs(x1-x2) < tolerance and y1 >= y2 and predicted_class1 == 'PLUS' and predicted_class2 == 'MINUS':
        return 'DOWN'
    
    if abs(x1-x2) < tolerance and y1 <= y2 and predicted_class1 == 'PLUS' and predicted_class2 == 'MINUS':
        return 'UP'
    
    if abs(y1-y2) < tolerance and x1 >= x2 and predicted_class1 == 'MINUS' and predicted_class2 == 'PLUS':
        return 'LEFT'
    if abs(y1-y2) < tolerance and x1 <= x2 and predicted_class1 == 'MINUS' and predicted_class2 == 'PLUS':
        return 'RIGHT'
    if abs(x1-x2) < tolerance and y1 >= y2 and predicted_class1 == 'MINUS' and predicted_class2 == 'PLUS':
        return 'UP'
    if abs(x1-x2) < tolerance and y1 <= y2 and predicted_class1 == 'MINUS' and predicted_class2 == 'PLUS':
        return 'DOWN'
    
    
    
    
    

    
    
# print(get_voltage_direction('dc_source.png'))
    


















