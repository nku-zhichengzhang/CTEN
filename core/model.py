import torch.nn as nn
from models.vaanet import VAANet
from models.visual_stream import VisualStream
#from models.visual_stream_gyc import VisualStream as VisualErase
from models.visual_erase_gyc import VisualErase
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def generate_model(opt):
    model = VAANet(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained,
    )
    model = model.cuda()
    return model, model.parameters()

def generate_visual_model(opt):
    model=VisualStream(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        pretrained_resnet101_path=opt.resnet101_pretrained,
    )
    model = nn.DataParallel(model)
    model=model.cuda()
    return model,model.parameters()

def generate_visual_Erase_model(opt):
    model=visual_stream_w_Erase(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        pretrained_resnet101_path=opt.resnet101_pretrained,
    )
    model = nn.DataParallel(model)
    model=model.cuda()
    return model, model.parameters()
