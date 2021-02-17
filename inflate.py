import clip, network, torch, video_loader as vl

# setup
# device = torch.device("cuda:0")
# res_net3D = network.generate_model(50, n_classes=700)
# criterion = vl.CrossEntropyLoss().to(device)
# res_net3D.load_state_dict(torch.load('./RN50.pt')['state_dict'])
# res_net3D.fc = vl.nn.Linear(512 * 4, 45)
# optimizer = vl.SGD(res_net3D.parameters(), lr=0.001, momentum=0.9)
# res_net3D.to(device)

