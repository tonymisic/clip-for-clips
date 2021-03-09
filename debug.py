import clip, network, torch, video_loader as vl, pickle, numpy as np, math
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
# load models
our_model = network.generate_model(50)
# our_model.load_state_dict(torch.load('./3d3N50_CLIP.pth'))
CLIP, preprocess = clip.load("RN50", device=device)
CLIP.visual = our_model
CLIP.type(torch.FloatTensor).to(device)
# our implementation of the loss
def clip_loss(video_feature_batch, text_feature_batch, batch_size, temperature):
    # l2 normalization
    video_feature_batch = video_feature_batch / video_feature_batch.norm(dim=-1, keepdim=True)
    text_feature_batch = text_feature_batch / text_feature_batch.norm(dim=-1, keepdim=True)
    # cosine sim as logits
    logit_scale = math.exp(temperature)
    logits_per_video = logit_scale * video_feature_batch @ text_feature_batch.t()
    logits_per_text = logit_scale * text_feature_batch @ video_feature_batch.t()
    # cross entropy loss
    labels = torch.eye(batch_size).type(torch.LongTensor).to(device)
    #### Doesn't work
    loss = torch.nn.CrossEntropyLoss()
    label = torch.Tensor([1.]).type(torch.LongTensor).to(device)
    video_loss = loss(logits_per_video, label)
    text_loss = loss(logits_per_text, labels[:, 0:])
    overall_loss = (video_loss + text_loss) / 2
    #### Doesn't work
    return video_loss, text_loss, overall_loss

# load example video and text
video = torch.from_numpy(vl.norm(vl.random_temporal_crop(vl.center_spatial_crop(vl.read_video('dog.mp4'), 224, 224), 64)))
video = video.unsqueeze_(0)
video = video.permute([0,4,1,2,3]).type(torch.half).to(device)
text = clip.tokenize(["a dog"]).to(device)
with torch.no_grad():
    video_features = CLIP.encode_video(video)
    text_features = CLIP.encode_text(text)
    logits_per_image, logits_per_text = CLIP(video, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

clip_loss(video_features, text_features, 1, 2)
# debug our model
# model = network.generate_model(50)
# model.load_state_dict(torch.load('./3dRN50_CLIP.pth'))
# x = torch.rand(1,3,64,224,224) # batch, chann, time, hieght, weidth
# o = model(x)
# print(o.size())



