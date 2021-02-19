import clip, network, torch, video_loader as vl, pickle, numpy as np

a = torch.Tensor([[0,1,2],[3,4,5],[6,7,8]])
a.unsqueeze_(-1)
a = a.expand(a.size())
b,c,d = a.size()
print(b,c,d)
print(a.size())
print(a)
exit()


ResNet503D_model = network.generate_model(50)
clip_model = torch.jit.load('./RN50.pt')

for i, var_name in enumerate(clip_model.state_dict()):
    current_tensor, current_3D_tensor = None, None
    if 'visual' in var_name and len(var_name.split('.', 1)) > 1: 
        current_3D_name = var_name.split('.', 1)[1]
        if current_3D_name in ResNet503D_model.state_dict():
            print("Key: " + var_name +  " found ... attempting copy")
            current_tensor, current_3D_tensor = clip_model.state_dict()[var_name], ResNet503D_model.state_dict()[current_3D_name]
            if current_3D_tensor.size() != current_tensor.size():
                print(current_3D_tensor.size())
                print(current_tensor.size())
                inflated = current_tensor.unsqueeze_(-1)
                inflated = inflated.expand()
                ResNet503D_model.state_dict()[current_3D_name] = inflated
                print("Successfully inflated and copied tensor!")
            else:
                ResNet503D_model.state_dict()[current_3D_name] = clip_model.state_dict()[var_name]
                print("Successfully copied same sized tensor!")
        else:
            print("Key: " + var_name +  " NOT found!")
    else:
        print("Key " + var_name + " not in visual part of CLIP!")

torch.save(ResNet503D_model.state_dict(), './3d3N50_CLIP.pth')
