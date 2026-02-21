import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G / (c * h * w)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

cnn = models.vgg19(pretrained=True).features.to(device).eval()

content_layers = ['conv_4']
style_layers = ['conv_1','conv_2','conv_3','conv_4','conv_5']

def get_model_and_losses(style_img, content_img):
    cnn_copy = copy.deepcopy(cnn)
    content_losses = []
    style_losses = []
    model = nn.Sequential()
    i = 0

    for layer in cnn_copy.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_"+str(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_"+str(i), style_loss)
            style_losses.append(style_loss)

    return model, style_losses, content_losses

def run_style_transfer(content_path, style_path, output_path, steps=150):
    content_img = load_image(content_path)
    style_img = load_image(style_path)
    input_img = content_img.clone()

    model, style_losses, content_losses = get_model_and_losses(style_img, content_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    run = [0]
    while run[0] <= steps:

        def closure():
            input_img.data.clamp_(0,1)
            optimizer.zero_grad()
            model(input_img)

            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)

            loss = 1e6 * style_score + content_score
            loss.backward()
            run[0] += 1
            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0,1)
    output = input_img.cpu().clone().squeeze(0)
    output = transforms.ToPILImage()(output)
    output.save(output_path, format="JPEG")