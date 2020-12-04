# from keras.models import Model
# from keras.optimizers import Adam
# from keras import backend as K
import data, model
from torch.optim import Adam

# smooth = 1.
# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

# def dice_coef_loss(y_true, y_pred):
#     return 1. - dice_coef(y_true, y_pred)

def unet_train(unet, dataloader, epoch = 3):
    # unet.cuda()
    unet.train()
    opt = Adam(unet.parameters())

    for i in range(epoch):
        for step, [batch_x, batch_y] in enumerate(dataloader):
            # print("step:{}, batch_x:{}, batch_y:{}".format(step, batch_x.shape, batch_y.shape))
            pass
        pass

    pass

if __name__ == "__main__":
    unet_train(model.UNet(), data.train_data_load())
    pass