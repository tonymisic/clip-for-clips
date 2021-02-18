import clip, network, torch, video_loader as vl, pickle, numpy as np

def show_clip_state():
    model = torch.jit.load('./RN50.pt')
    print("CLIP's RN50 state_dict:")
    for var_name in model.state_dict():
        print(var_name, "\t", model.state_dict()[var_name].size())

model = network.generate_model(50)
print("Random 3dRN50 state_dict:")
for var_name in model.state_dict():
    print(var_name, "\t", model.state_dict()[var_name].size())


