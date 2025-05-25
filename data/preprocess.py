import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

def preprocess_image(image):
    # Convert PIL Image to numpy array (for cv2)
    img = np.array(image.convert('L'))  # convert to grayscale

    # Resize to 256x256
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    # Median blur denoising
    img = cv2.medianBlur(img, 3)

    # Adaptive thresholding (binary inverse)
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )

    # Convert back to PIL Image
    pil_img = Image.fromarray(binary).convert('L')

    # Final transforms: resize 128x128, ToTensor, normalize
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # grayscale normalization
    ])

    return transform(pil_img)
