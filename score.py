import torch
from PIL import Image
from torchvision import transforms
import os


def score(images, model_ptl):
    model = torch.jit.load(model_ptl)
    model.eval()
    output = {}

    for image in images:
        input_image = Image.open(image).convert('RGB')
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output.update({os.path.basename(image): model(input_batch)})

    return output


if __name__ == "__main__":
    images = ['images/Yes_1.png', 'images/No_1.png', 'images/Yes_2.png', 'images/No_2.png']
    print(score(images, 'data/model.ptl'))
