from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy
import os
import time
import torch
from datasets.ve8 import VE8Dataset
from torch.utils.data import DataLoader

def get_ve8(video_path,audio_path,annotation_path, subset, transforms):
    spatial_transform, temporal_transform, target_transform = transforms
    return VE8Dataset(video_path,
                      audio_path,
                      annotation_path,
                      subset,
                      30,
                      spatial_transform,
                      temporal_transform,
                      target_transform,
                      need_audio=False)

def get_validation_set(dataset,video_path,audio_path,annotation_path, spatial_transform, temporal_transform, target_transform):
    if dataset == 've8':
        transforms = [spatial_transform, temporal_transform, target_transform]
        return get_ve8(video_path,audio_path,annotation_path, 'validation', transforms)
    else:
        raise Exception

def test_vaanet(epoch, data_loader, model, criterion, opt, writer, optimizer):
    print("# ---------------------------------------------------------------------- #")
    print('Test our model'.format(epoch))
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, data_item in enumerate(data_loader):
        visual, target, audio, visualization_item, batch_size = process_data_item(opt, data_item)
        data_time.update(time.time() - end_time)
        with torch.no_grad():
            output, loss = run_model(opt, [visual, target, audio], model, criterion, i)
        acc = calculate_accuracy(output, target)
        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)
        batch_time.update(time.time() - end_time)
        end_time = time.time()
    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/acc', accuracies.avg, epoch)
    print("Val loss: {:.4f}".format(losses.avg))
    print("Val acc: {:.4f}".format(accuracies.avg))
    save_file_path = os.path.join(opt.ckpt_path, 'save_{}_{:.4f}.pth'.format(epoch,accuracies.avg))
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_file_path)

if __name__ == "__main__":
    dataset='ve8'
    video_path="/home/ubuntu7/wlj/dataset/Youtube-8-jpg"
    audio_path="VideoEmotion8--mp3"
    annotation_path="/home/ubuntu7/wlj/code/VAANet-master/data/k_fold/ve8_05.json"
    