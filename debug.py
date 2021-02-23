import clip, network, torch, video_loader as vl, pickle, numpy as np
from PIL import Image

# debug clip
device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP, preprocess = clip.load("RN50", device=device)
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
with torch.no_grad():
    image_features = CLIP.encode_image(image)
    text_features = CLIP.encode_text(text)
    logits_per_image, logits_per_text = CLIP(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
print("Label probs:", probs) 

# debug our model
new_lad = torch.load('./3d3N50_CLIP.pth')
model = network.generate_model(50)
model.load_state_dict(new_lad)
x = torch.rand(1,3,64,224,224) # batch, chann, time, hieght, weidth
o = model(x)
print(o)