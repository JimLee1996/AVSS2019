import time
import torch
from utils import AverageMeter


def train(
    epoch, data_loader, model, criterion, optimizer, device, batch_log,
    epoch_log
):
    print('training at epoch: {}'.format(epoch))

    # meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # set model to training mode
    model.train()

    end_time = time.time()

    for i, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        data_time.update(time.time() - end_time)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        # meter
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        # backward + optimize
        loss.backward()
        optimizer.step()

        # meter
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print(
            'Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch,
                i + 1,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                acc=accuracies
            )
        )

        batch_log.log(
            {
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'acc': accuracies.val,
                'lr': optimizer.param_groups[0]['lr']
            }
        )

    epoch_log.log(
        {
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': optimizer.param_groups[0]['lr']
        }
    )


def val(epoch, data_loader, model, criterion, device, val_log):
    print('validation at epoch: {}'.format(epoch))

    # set model to evaluate mode
    model.eval()

    # meters
    losses = AverageMeter()
    accuracies = AverageMeter()

    for _, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # no need to track grad in eval mode
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

    print(
        'Epoch: [{}]\t'
        'Loss(val): {loss.avg:.4f}\t'
        'Acc(val): {acc.avg:.3f}'.format(epoch, loss=losses, acc=accuracies)
    )

    val_log.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    return losses.avg, accuracies.avg


def test():
    pass


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size
