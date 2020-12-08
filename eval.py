import torch, os
from loss import DiceLoss
from PIL import Image
import numpy as np

def eval(state, dataloader, epoch = 3, check_points_path = 'checkpoints', save_path = os.getcwd()):
    for i in range(epoch):
        save_img_path = f'{os.path.join(os.getcwd(), check_points_path)}/epoch{i}'
        try:
            os.mkdir(save_img_path}
            pass
        except:
            pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(os.path.join(check_points_path, f'epoch{i}.pth'))
        unet = state[net].double().to(device)
        opt = state[opt]
        unet.eval()
        loss = DiceLoss()

        average_loss = 0.
        for step, [batch_x, batch_y] in enumerate(dataloader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = unet(batch_x)
            for j in range(y_pred.shape[0]):
                img = y_pred[j, 0, :, :].data.cpu()
                img = Image.fromarray(img)
                img.save(f'{save_img_path}/{i * len(dataloader) + j}.jpg', quality=95)
                np.savetxt(f'{save_img_path}/{i * len(dataloader) + j}', img)
                pass
            dice_loss = loss(batch_y, y_pred)
            print(f'dice loss in step {step}: {dice_loss}')
            average_loss = average_loss + dice_loss
            pass
        average_loss = average_loss / len(dataloader)
        print(f'average loss in epoch {i}: {average_loss}')
        pass
    pass