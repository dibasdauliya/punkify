import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.models import vgg19
from PIL import Image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations for input frames
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define inverse transform to convert back to image format
inv_transform = transforms.Compose([
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.ToPILImage(),
])

# Load pre-trained VGG model (or a Fast Neural Style Transfer model)
class FastStyleTransferModel(torch.nn.Module):
    def __init__(self, style_path):
        super(FastStyleTransferModel, self).__init__()
        self.vgg = vgg19(pretrained=True).features[:21].to(device).eval()
        # Load a pre-trained style image model if available
        # Placeholder for simplicity - load actual trained model here
        self.style_image = transform(Image.open(style_path).convert('RGB')).unsqueeze(0).to(device)

    def forward(self, x):
        # Apply style transfer here
        # This placeholder does not modify the input; replace with actual style transfer logic
        return x

# Initialize style transfer model
model = FastStyleTransferModel("path_to_your_steampunk_style_image.jpg").to(device)

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Main loop to capture and style frames in real-time
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Transform frame and send through model
    frame_tensor = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        styled_frame_tensor = model(frame_tensor)

    # Convert the styled frame back to image format
    styled_frame = styled_frame_tensor.squeeze(0).cpu()
    styled_frame = inv_transform(styled_frame)
    styled_frame = np.array(styled_frame)
    
    # Display the result
    cv2.imshow("Steampunk Style Live Feed", cv2.cvtColor(styled_frame, cv2.COLOR_RGB2BGR))

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
