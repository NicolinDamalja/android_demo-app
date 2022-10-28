import torch
from PIL import Image
from torchvision import transforms
import os
import model


def score(image, model_ptl):
    model_ = model.model_static(model_ptl)
    model_dict = model_.state_dict()
    snapshot = torch.load(model_ptl)
    model_dict.update(snapshot)
    model_.load_state_dict(model_dict)

    model_.train(False)
    #model.eval()
    output_ = {}

    input_image = Image.open(image).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    output = model_(input_batch)
    with torch.no_grad():
        score = torch.sigmoid(output).item()
        output_.update({os.path.basename(image): score})
        #output.update({os.path.basename(image): model_(input_batch)})

    return output_


if __name__ == "__main__":
    images = ['images/No_1.png', 'images/No_2.png', 'images/Yes_1.png', 'images/Yes_2.png']
    for image in images:
        print(score(image, 'data/model_weights.pkl'), "\n")
