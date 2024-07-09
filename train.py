from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy,batch_augment,batch_augment2
import time
import torch
from tqdm import tqdm

def train_epoch(epoch, data_loader, model, criterion, optimizer, opt, class_names, writer):
    print("# ---------------------------------------------------------------------- #")
    print('Training at epoch {}'.format(epoch))
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1=AverageMeter()
    losses2=AverageMeter()
    losses3=AverageMeter()
    accuracies1 = AverageMeter()
    accuracies2=AverageMeter()
    accuracies3 = AverageMeter()
    end_time = time.time()
    for i, data_item in enumerate(data_loader):
        visual, target, audio, visualization_item, batch_size, video_item = process_data_item(opt, data_item)
        data_time.update(time.time() - end_time)
        output1, loss1, gamma1 = run_model(opt, [visual, target, audio], model, criterion, i, print_attention=False)
        gamma_row_max=torch.max(gamma1,dim=1)[0]*0.7 + torch.min(gamma1,dim=1)[0]*0.3
        gamma_row_max=gamma_row_max.unsqueeze(0).transpose(1,0)
        gamma_thre=gamma_row_max.expand(gamma1.shape)
        high_index=gamma1<gamma_thre
        low_index=gamma1>gamma_thre
        visual_erase2=batch_augment(video_item,high_index,opt,visual)
        output2, loss2, gamma2 = run_model(opt, [visual_erase2, target, audio], model, criterion, i, print_attention=False)
        visual_erase3=batch_augment2(video_item, low_index, opt, visual)
        output3, loss3, gamma3 = run_model(opt, [visual_erase3, target, audio], model, criterion, i, print_attention=False)
        loss=loss1/3.+loss2/3.+loss3/3.
        acc1 = calculate_accuracy(output1, target)
        acc2 = calculate_accuracy(output2,target)
        acc3 = calculate_accuracy(output3,target)
        losses.update(loss.item(), batch_size)
        losses1.update(loss1.item(),batch_size)
        losses2.update(loss2.item(),batch_size)
        losses3.update(loss3.item(),batch_size)
        accuracies1.update(acc1, batch_size)
        accuracies2.update(acc2, batch_size)
        accuracies2.update(acc3, batch_size)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        iter = (epoch - 1) * len(data_loader) + (i + 1)
        writer.add_scalar('train/batch/loss', losses.val, iter)
        writer.add_scalar('train/batch/acc1', accuracies1.val, iter)
        writer.add_scalar('train/batch/acc2', accuracies2.val, iter)
        if opt.debug:
            H, L = high_index.sum(), low_index.sum()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                  'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                  'Acc1 {acc.val:.3f} ({acc.avg:.3f})\t'
                  'Acc2 {acc2.val:.3f} ({acc2.avg:.3f})\t'
                  'Acc3 {acc3.val:.3f} ({acc3.avg:.3f})\t'
                  'Erase {H:.3f} ({L:.3f})\t'.format(
                epoch, i + 1, len(data_loader), batch_time=batch_time, data_time=data_time, loss=losses,loss1=losses1,loss2=losses2, acc=accuracies1, acc2=accuracies2,acc3=accuracies3, H=H, L=L))
    # ---------------------------------------------------------------------- #
    print("Epoch Time: {:.2f}min".format(batch_time.avg * len(data_loader) / 60))
    print("Train loss: {:.4f}".format(losses.avg))
    print("Train acc1: {:.4f}".format(accuracies1.avg))
    print("Train acc2: {:.4f}".format(accuracies2.avg))
    print("Train acc3: {:.4f}".format(accuracies3.avg))
    writer.add_scalar('train/epoch/loss', losses.avg, epoch)
    writer.add_scalar('train/epoch/acc', accuracies1.avg, epoch)
    writer.add_scalar('train/epoch/acc', accuracies2.avg, epoch)
