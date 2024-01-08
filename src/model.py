import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics


class neural_network_fits(pl.LightningModule):
    def __init__(self, initial_layer_size=1280, lr=1e-3):
        """ Given an input esm2 sequence embedding, 
        predicts the mean and sigma (standard deviation) parameters e.g., for pH activity curves.
        """
        super(neural_network_fits, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(initial_layer_size, 512),
            nn.LeakyReLU(),            
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 2),
        )
        self.lr = lr
        # l1 loss reduction
        self.reduction = "mean"

    def forward(self, x):
        # predict the mean and std deviation
        logits = self.linear_relu_stack(x)
        #return logits
        # y_pred_mean, y_pred_sigma = logits[:,0], logits[:,1]

        # take the sigmoid of the pH mean logits, then multiply by 13 to get in the range 0-13
        y_pred_mean = torch.sigmoid(logits[:,0])
        y_pred_mean = torch.multiply(y_pred_mean, 13)
        # limit the standard deviation to between 0 and 4.25
        y_pred_sigma = torch.sigmoid(logits[:,1])
        y_pred_sigma = torch.multiply(y_pred_sigma, 4.25)
        return y_pred_mean, y_pred_sigma

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_opt, y_std = y[:,0], y[:,1]
        
        logits = self(x)
        y_pred_mean, y_pred_sigma = logits[:,0], logits[:,1]
        #y_pred_mean, y_pred_sigma = self(x)

        # use mse_loss or l1_loss (mae) on both the ph opt
        loss = nn.functional.l1_loss(y_pred_mean, y_opt, reduction=self.reduction)
        # and the ph standard deviation. Only use rows that are not nan
        loss += nn.functional.l1_loss(y_pred_sigma[~torch.isnan(y_std)], 
                                      y_std[~torch.isnan(y_std)],
                                      reduction=self.reduction)
        self.log('train_loss', loss)
        return loss
    
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_ph, y_act = y[:,0], y[:,1]
        
#         y_pred = self(x)
#         y_pred_mean, y_pred_sigma = y_pred[:,0], y[:,1]
#         y_pred_act = self.get_gaus_pred(y_ph, y_pred_mean, y_pred_sigma)

#         loss = nn.functional.l1_loss(y_pred_act, y_act)
#         self.log("val_loss", loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
