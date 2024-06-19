import torch
import torch.utils.data as data

from torchvision import get_image_backend

from PIL import Image
import random
import json
import os
import functools
# import librosa
import numpy as np
import torchaudio

def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        return float(input_file.read().rstrip('\n\r'))

def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)

def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []
    for key, value in data['database'].items():
        if value['subset'] == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])
    return video_names, annotations

def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def get_default_image_loader():
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))
        assert os.path.exists(image_path), image_path + " image does not exists"
        video.append(image_loader(image_path))
    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)

def preprocess_audio(audio_path):
    "Extract audio features from an audio file"
    y, sr = librosa.load(audio_path, sr=44100)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)
    return mfccs

class VE8Dataset(data.Dataset):
    def __init__(self,
                 video_path,
                 audio_path,
                 annotation_path,
                 subset,
                 fps=30,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 get_loader=get_default_video_loader,
                 need_audio=True):
        self.data, self.class_names = make_dataset(
            video_root_path=video_path,
            annotation_path=annotation_path,
            audio_root_path=audio_path,
            subset=subset,
            fps=fps,
            need_audio=need_audio
        )
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.fps = fps
        self.ORIGINAL_FPS = 30
        self.need_audio = need_audio
        self.norm_mean = -6.6268077
        self.norm_std = 5.358466
        self.audio_n_segments = 16

    def __getitem__(self, index):
        data_item = self.data[index]
        video_path = data_item['video']
        frame_indices = data_item['frame_indices']
        snippets_frame_idx = self.temporal_transform(frame_indices)
        if self.need_audio:
            timeseries_length = 100*self.audio_n_segments
            waveform, sr = torchaudio.load(data_item['audio'])
            waveform = waveform - waveform.mean()
            fbank = torchaudio.compliance.kaldi.fbank(waveform, 
                                                htk_compat=True,
                                                sample_frequency=sr,
                                                use_energy=False,
                                                window_type='hanning',
                                                num_mel_bins=128,
                                                dither=0.0,
                                                frame_shift=10)
            if fbank.shape[0]<=timeseries_length:
                k = timeseries_length // fbank.shape[0] + 1
                fbank = np.tile(fbank, reps=(k, 1))
                audios = fbank[:timeseries_length, :]
            else:
                blk = int(fbank.shape[0]/self.audio_n_segments)
                aud = []
                for i in list(range(0,fbank.shape[0],blk))[:self.audio_n_segments]:
                    ind = i+int(random.random()*(blk-100))
                    aud.append(fbank[ind:ind+100])
                audios = torch.cat(aud)
            if audios.shape[0]!=timeseries_length:
                print(audios.shape)
            audios = torch.FloatTensor(audios)
            audios = (audios - self.norm_mean) / (self.norm_std * 2)
        else:
            audios = []
        snippets = []
        for snippet_frame_idx in snippets_frame_idx:
            snippet = self.loader(video_path, snippet_frame_idx)
            snippets.append(snippet)
        self.spatial_transform.randomize_parameters()
        snippets_transformed = []
        for snippet in snippets:
            snippet = [self.spatial_transform(img) for img in snippet]
            snippet = torch.stack(snippet, 0).permute(1, 0, 2, 3)
            snippets_transformed.append(snippet)
        snippets = snippets_transformed
        snippets = torch.stack(snippets, 0)
        target = self.target_transform(data_item)
        visualization_item = [data_item['video_id']]
        return snippets, target, audios, visualization_item, data_item['video'], data_item['n_frames']

    def __len__(self):
        return len(self.data)

def make_dataset(video_root_path, annotation_path, audio_root_path, subset, fps=30, need_audio=True):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name
    dataset = []
    for i in range(len(video_names)):
        if i % 100 == 0:
            print("Dataset loading [{}/{}]".format(i, len(video_names)))
        video_path = os.path.join(video_root_path, video_names[i])
        if not os.path.exists(video_path):
            print(video_path)
            continue
        if need_audio:
            audio_path = os.path.join(audio_root_path, video_names[i] + '.mp3')
            if not os.path.exists(audio_path):
                print(audio_path)
                continue
        else:
            audio_path = None
        
        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            print(video_path)
            continue
        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1],
        }
        if need_audio: sample['audio'] = audio_path
        assert len(annotations) != 0
        sample['label'] = class_to_idx[annotations[i]['label']]
        ORIGINAL_FPS = 30
        step = ORIGINAL_FPS // fps
        sample['frame_indices'] = list(range(1, n_frames + 1, step))
        dataset.append(sample)
    return dataset, idx_to_class
