import clip, network, torch, video_loader as vl, pickle, numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
# load models
our_model = network.generate_model(50)
# our_model.load_state_dict(torch.load('./3d3N50_CLIP.pth'))
CLIP, preprocess = clip.load("RN50", device=device)
CLIP.visual = our_model
# load example video and text
video = torch.from_numpy(vl.norm(vl.random_temporal_crop(vl.center_spatial_crop(vl.read_video('dog.mp4'), 224, 224), 64))).float()
video = video.unsqueeze_(0)
video = video.permute([0,4,1,2,3])
text = clip.tokenize(["a dog", "a store"]).to(device)
with torch.no_grad():
    video_features = CLIP.encode_video(video)
    text_features = CLIP.encode_text(text)
    logits_per_image, logits_per_text = CLIP(video, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# debug our model
# model = network.generate_model(50)
# model.load_state_dict(torch.load('./3dRN50_CLIP.pth'))
# x = torch.rand(1,3,64,224,224) # batch, chann, time, hieght, weidth
# o = model(x)
# print(o.size())