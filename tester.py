import clip, network, torch, video_loader as vl, pickle, numpy as np
from PIL import Image

new_lad = torch.load('./3d3N50_CLIP.pth')
model = network.generate_model(50)
model.load_state_dict(new_lad)
# debugger
x = torch.rand(1,3,64,224,224) # batch, chann, time, hieght, weidth
o = model(x)
print(o)