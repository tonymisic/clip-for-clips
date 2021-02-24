import clip, network, torch, video_loader as vl, pickle, numpy as np

new_lad = torch.load('./3d3N50_CLIP.pth')
clip_model = torch.jit.load('./RN50.pt')
ResNet503D_model = network.generate_model(50)

with torch.no_grad():
    # iterate through all clip key-value pairs
    for i, var_name in enumerate(clip_model.state_dict()):
        # intitalize tensors
        current_tensor, current_3D_tensor = None, None
        # CLIP's ResNet is denoted with "visual."
        if 'visual' in var_name and len(var_name.split('.', 1)) > 1: 
            # remove "visual."
            current_3D_name = var_name.split('.', 1)[1]
            # find key in regular 3DResNet
            if current_3D_name in ResNet503D_model.state_dict():
                print("Key: " + var_name +  " found ... attempting copy")
                # get associated tensors
                current_tensor = clip_model.state_dict()[var_name].detach().clone()
                current_3D_tensor = ResNet503D_model.state_dict()[current_3D_name].detach().clone()
                # check size difference
                if current_3D_tensor.size() != current_tensor.size():
                    # inflation across time dimension
                    print("3D Tensor: ", current_3D_tensor.size())
                    print("CLIP: ", current_tensor.size())
                    inflated = current_tensor.unsqueeze_(-3)
                    a,b,c,d,e = current_3D_tensor.size()
                    inflated = inflated.expand(a,b,c,d,e)
                    print("Inflated: ", inflated.size())
                    ResNet503D_model.state_dict()[current_3D_name].copy_(inflated)
                    print("Successfully inflated and copied tensor!")
                else:
                    ResNet503D_model.state_dict()[current_3D_name].copy_(clip_model.state_dict()[var_name]) # copy values if same sized
                    print("Successfully copied same sized tensor!")
            else:
                print("Key: " + var_name +  " NOT found!")
        else:
            print("Key " + var_name + " not in visual part of CLIP!")
torch.save(ResNet503D_model.state_dict(), './3d3N50_CLIP.pth')
