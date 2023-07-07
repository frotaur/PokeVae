import torchvision.utils as tuts, torch
import torch.optim.lr_scheduler as lrsched
from torch.optim import Optimizer
from torchenhanced import Trainer
from VAE import VAE
from torch.utils.data import DataLoader


class TrainerVae(Trainer):

    def __init__(self, model: VAE,dataset, optim: Optimizer = None, scheduler: lrsched._LRScheduler = None, model_save_loc=None, state_save_loc=None, device='cpu', run_name=None):
        super().__init__(model, optim, scheduler, model_save_loc, state_save_loc, device, run_name)
        self.dataset = dataset

    def process_batch(self, batch_data, data_dict: dict, **kwargs):
        img_batch = batch_data
        batch_log = data_dict['batch_log']

        out, (loss_mse,loss_kull) = self.model(img_batch)

        with torch.no_grad():
            if(data_dict['batchnum']%batch_log==batch_log-1):
                predvstrue = tuts.make_grid(torch.cat([out[:4],img_batch[:4]]))
                self.writer.add_image('predicted vs true',predvstrue.detach().cpu(),data_dict['time'])
                self.writer.add_scalar('kulloss',loss_kull.item(),data_dict['time'])
                self.writer.add_scalar('loss_mse',loss_mse.item(),data_dict['time'])
        
        return loss_mse+loss_kull, data_dict
    
    def process_batch_valid(self, batch_data, data_dict: dict, **kwargs):
        pass
    
    def change_lr(self,lr):
        for g in self.optim.param_groups:
            g['lr'] = lr
            
    def get_loaders(self, batch_size):
        loader = DataLoader(self.dataset,batch_size=batch_size,shuffle=True)

        return (loader,None)