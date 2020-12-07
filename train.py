import data, model
import os
import torch
from torch.optim import Adam
from torch.nn import Module


class DiceLoss(Module):
    def __init__(self, eps = 1e-5):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, y_true, y_pred):
        N = y_pred.shape[0]
        y_true_f = y_true.view(N, -1)
        y_pred_f = y_pred.view(N, -1)
        tp = torch.sum(y_true_f * y_pred_f, dim=1)
        fp = torch.sum(y_pred_f, dim=1) - tp
        fn = torch.sum(y_true_f, dim=1) - tp
        dice_coef = (2. * tp + self.eps) / (2. * tp + fp + fn + self.eps)
        return torch.mean(1. - dice_coef)


def unet_train(unet, dataloader, epoch = 3, check_points_path = 'checkpoints', save_path = "."):
    unet = unet.double()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = unet.to(device)
    unet.train()
    opt = Adam(unet.parameters())
    loss = DiceLoss()

    for i in range(epoch):
        epoch_loss = 0.
        for step, [batch_x, batch_y] in enumerate(dataloader):
            print("step:{}, batch_x:{}, batch_y:{}".format(step, batch_x.shape, batch_y.shape))
            batch_x = batch_x.to(device, dtype=torch.float64)
            batch_y = batch_y.to(device, dtype=torch.long)

            y_pred = unet(batch_x)
            # print(y_pred.shape)
            dice_loss = loss(batch_y, y_pred)
            print(dice_loss)
            epoch_loss = epoch_loss + dice_loss.item()

            opt.zero_grad()
            dice_loss.backward()
            opt.step()

            pass
        epoch_loss = epoch_loss / len(dataloader)
        print(f'average loss: {epoch_loss}')

        if check_points_path == None:
            continue
        
        try:
            os.mkdir(check_points_path)
        except expression as identifier:
            pass
        state = {
            'net' : unet.state_dict(),
            'opt' : opt.state_dict(),
            'epoch' : i
        }
        torch.save(state, os.path.join(check_points_path, f'epoch{i}.pth'))
        pass
    
    torch.save(unet.state_dict(), os.path.join(save_path, 'net.pth'))

    pass

if __name__ == "__main__":
    unet_train(model.UNet(), data.train_data_load())
    pass
