# from keras.models import Model
# from keras.optimizers import Adam
# from keras import backend as K
import data, model
import torch
from torch.optim import Adam
from torch.nn import Module

# smooth = 1.
# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

# def dice_coef_loss(y_true, y_pred):
#     return 1. - dice_coef(y_true, y_pred)

# def dice_coef(y_true, y_pred, eps = 1e-5):
#     N = y_pred.shape[0]
#     y_true_f = y_true.view(N, -1)
#     y_pred_f = y_pred.view(N, -1)
#     tp = torch.sum(y_true_f * y_pred_f, dim=1)
#     fp = torch.sum(y_pred_f, dim=1) - tp
#     fn = torch.sum(y_true_f, dim=1) - tp
#     return (2. * tp + eps) / (2. * tp + fp + fn + eps) / N

# def dice_coef_loss(y_true, y_pred, eps = 1e-5):
#     return 1. - dice_coef(y_true, y_pred, eps)

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

# class SoftDiceLossV2(_Loss):
#     __name__ = 'dice_loss'
 
#     def __init__(self, reduction='mean'):
#         super(SoftDiceLossV2, self).__init__()
 
#     def forward(self, y_pred, y_true):
#         class_dice = []
#         for i in range(1, self.num_classes):
#             class_dice.append(diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :]))
#         mean_dice = sum(class_dice) / len(class_dice)
#         return 1 - mean_dice

# class DiceLoss(Module):

#     def __init__(self) -> None:
#         super(DiceLoss, self).__init__()
#         self.eps: float = 1e-6

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         if not torch.is_tensor(input):
#             raise TypeError("Input type is not a torch.Tensor. Got {}"
#                             .format(type(input)))
#         if not len(input.shape) == 4:
#             raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
#                              .format(input.shape))
#         if not input.shape[-2:] == target.shape[-2:]:
#             raise ValueError("input and target shapes must be the same. Got: {}"
#                              .format(input.shape, input.shape))
#         if not input.device == target.device:
#             raise ValueError(
#                 "input and target must be in the same device. Got: {}" .format(
#                     input.device, target.device))

#         # create the labels one hot tensor
#         target_one_hot = one_hot(target, num_classes=input.shape[1],
#                                  device=input.device, dtype=input.dtype)

#         # compute the actual dice score
#         dims = (1, 2, 3)
#         intersection = torch.sum(input_soft * target_one_hot, dims)
#         cardinality = torch.sum(input_soft + target_one_hot, dims)

#         dice_score = 2. * intersection / (cardinality + self.eps)
#         return torch.mean(1. - dice_score)


def unet_train(unet, dataloader, epoch = 3):
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
        print(epoch_loss / len(dataloader))
        pass

    pass

if __name__ == "__main__":
    unet_train(model.UNet(), data.train_data_load())
    pass
