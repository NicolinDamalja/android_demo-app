import torch
from PIL import Image
from torchvision import transforms

def score(img, model_ptl ):
    model = torch.jit.load(model_ptl)
    model.eval()

    input_image = Image.open(img)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)[0]

    return output


if __name__ == "__main__":
    print(score('images/girl.jpeg', "data/model.ptl"))
