"""
Simple model for the diffusion time series.
"""

import torch
import torch.nn as nn

import lighting.pytorch as pl

class RNNModel(pl.LightningModule):
    """
    Simple RNN model.
    LSTM backbone
    """
    def __init__(self, dim_in=1, dim_out=1, hidden_size=32, dataset=None, delta_t=0.01):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden_size = hidden_size
        self.dataset = dataset
        self.delta_t = delta_t
        
        # one linear layer 
        self.linear = nn.Linear(dim_in, hidden_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # output layer
        self.output = nn.Linear(hidden_size, dim_out)
        
        # loss function
        self.loss = nn.MSELoss()
        
    def forward(self, x, y):
        """
        Forward pass of the model.
        """
        shape_x = x.shape
        shape_y = y.shape
        forcast_step = shape_y[1]
        
        # we concatenate the input
        x = torch.cat((x, y), dim=1)
        
        # first we apply the linear layer
        x = self.linear(x)
        
        # we apply the LSTM layer
        x, _ = self.lstm(x)
        
        # now we take only the last forcast_step element
        x = x[:, -forcast_step:, :]
        
        # we apply the output layer
        x = self.output(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        """
        Training step
        """
        y = batch["data_noisy"]
        gradiant = batch["gradiant"]
        x = batch["X"]
        t_value = batch["t_value"]
        
        gradiant_pred = self(x, y, t_value)
        
        loss = self.loss(gradiant_pred, gradiant)
        
        self.log("train_loss", loss)
        
        return loss
    
    def on_epoch_end(self):
        """
        Function called at the end of the epoch
        """
        self.eval()
        with torch.no_grad():
            x = torch.zeros((1, 4, self.dim_in))
            y = torch.randn(1, 4, self.dim_in) * torch.sqrt(
            torch.tensor(self.dataset.variance_values[-1])
        )
            
            for idx, time_step in enumerate(torch.flip(self.dataset.t_array, dims=[0])):
                t_value = (
                torch.tensor(time_step).unsqueeze(0)
                )  # should be a tensor of shape (1,)
                
                gradiant = self(x, y, t_value)
                
                beta_current = self.dataset.beta_values[99 - idx]
                
                noise = torch.randn_like(y)

                y = (
                y
                + beta_current * (0.5 * y + gradiant) * self.delta_t
                + torch.sqrt(torch.tensor(beta_current))
                * torch.sqrt(torch.tensor(self.delta_t))
                * noise
                )
                
                
            
    
    def configure_optimizers(self):
        """
        Configure the optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        
