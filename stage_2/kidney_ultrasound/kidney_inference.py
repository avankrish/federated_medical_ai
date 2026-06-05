import torch
from PIL import Image
from torchvision import transforms
from stage_2.kidney_ultrasound.model import KidneyCNN


model = KidneyCNN()
model.load_state_dict(torch.load("stage_2/kidney_ultrasound/kidney_model.pth"))
model.eval()


transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])


def kidney_inference(image_path):

    img = Image.open(image_path).convert("RGB")

    img = transform(img)

    img = img.unsqueeze(0)

    with torch.no_grad():

        output = model(img)

        prob = output.item()

        if prob > 0.5:
            result = "Kidney Abnormality Detected"
        else:
            result = "Normal Kidney"

    return {
        "probability": prob,
        "prediction": result
    }


if __name__ == "__main__":

    test_image = "stage_2/kidney_ultrasound/data/Stone/Stone_10.JPG"

    result = kidney_inference(test_image)

    print("\n===== KIDNEY ULTRASOUND INFERENCE =====")

    print("Prediction :", result["prediction"])
    print("Probability:", round(result["probability"],4))