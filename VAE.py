from torchenhanced import DevModule
import torch.nn as nn, torch


class Encoder(DevModule):
    """
        Encoder module. Given a batch of RGBA images, encodes them
        in the latent space as gaussians.
    """

    def __init__(self,img_size,latent_dim):
        super().__init__()
        self.img_size= img_size
        self.latent_dim = latent_dim

        self.convoluter = nn.Sequential(
            nn.Conv2d(4,16,kernel_size=(5,5)), # (16,img_s-4)
            nn.BatchNorm2d(16),
            nn.ReLU(), 
            nn.Conv2d(16,32,kernel_size=(3,3),stride=2,padding=1), # (32,img_s)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=(3,3),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=(3,3),stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,256,kernel_size=(3,3),stride=2,padding=1),
            nn.BatchNorm2d(256)
        )

        out_shape = self.comp_lin_chans()
        prod = 1
        for siz in out_shape :
            prod*=siz
        
        dense_in = prod
        self.to_latent = nn.Linear(dense_in,2*self.latent_dim)



    def forward(self,x) -> tuple[torch.Tensor]:
        """
            Computes the latent embedding distribution.

            params :
            x : (B,C,H,W) batch of images. Must have correct size as given by self.img_size

            returns :
            tuple : (mu,var_log), both of size (B,latent_dim)
        """
        B,_,_,_ = x.shape
        x = x.to(self.device)
        x = self.convoluter(x) # (B,C',H',W')

        x = torch.flatten(x,start_dim=1,end_dim=-1) # (B,C'*H'*W')

        mu,var_log = self.to_latent(x).split(self.latent_dim,dim=1)

        assert mu.shape==(B,self.latent_dim),f"Unexpected mu shape : {mu.shape}"

        return (mu, var_log)
    
    def sample(self,lat_mu,lat_logvar)-> torch.Tensor:
        """
            Sample a latent vector given the distribution induced by the image.

            params : 
            x: (B,C,H,W) batch of imgs.
            returns : tensor of shape (B,latent_dim) 
        """
        B,_ = lat_mu.shape
        epsilon = torch.randn((B,self.latent_dim))

        return lat_mu + torch.exp(lat_logvar/2.)*epsilon


    @torch.no_grad()
    def comp_lin_chans(self):
        """
            Use to compute shape of the final image, after convolutions.
        """
        dummy = torch.randn((2,4,*self.img_size)) # with batch size

        out = self.convoluter(dummy)
        print(f'Out of convo chans : {out.shape}')

        return out.shape[1:]


class Decoder(DevModule):
    """
        Encoder module. Given a batch of latent vectors, decodes them
        in a batch of images.
    """
    def __init__(self,final_img_size,latent_dim,start_img_size):
        super().__init__()
        self.img_size=final_img_size
        self.latent_dim=latent_dim

        self.start_img_size=start_img_size
        self.lat_to_img = nn.Linear(latent_dim,256*start_img_size[0]*start_img_size[1])

        # TEMP, does not work with any size
        self.deconvo = nn.ModuleList([
            nn.ConvTranspose2d(256,128,kernel_size=(3,3),stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,kernel_size=(3,3),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,kernel_size=(3,3),stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32,16,kernel_size=(3,3),stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16,4,kernel_size=(5,5)),
            nn.Sigmoid()
        ])

    def forward(self,latent_vec):
        B,_ = latent_vec.shape
        x = self.lat_to_img(latent_vec).reshape(-1,256,self.start_img_size[0],self.start_img_size[1])

        for lay in self.deconvo:
            x = lay(x)
        assert x.shape==(B,4,self.img_size[0],self.img_size[1]),f"Deconvolution produces incorrect size : {x.shape}"

        return x

class VAE(DevModule):
    """
        Autoencoder module.

        params :

    """

    def __init__(self,img_size,latent_dim, c_loss):
        super().__init__()
        self.encoder = Encoder(img_size,latent_dim)

        img_size_conv = self.encoder.comp_lin_chans()[1:] # take only img size, not channels.

        self.decoder = Decoder(final_img_size=img_size,latent_dim=latent_dim,start_img_size=img_size_conv)
        
        self.config = dict(img_size=img_size,latent_dim=latent_dim)

        self.c = c_loss

        self.gamma = nn.Parameter(torch.tensor(1.,device=self.device))

    def comp_loss(self,pred,target, lat_mu, lat_logvar):
        """
            Compute the VAE loss, including Kullback Leibler divergence
        """

        predloss =1./(2*(self.gamma)**2)*nn.functional.mse_loss(pred,target)*self.c+torch.log(self.gamma/torch.pi)
        kullbackloss = -.5*(lat_logvar+1-torch.exp(lat_logvar)-lat_mu**2)
        
        return predloss,kullbackloss.mean()

    def forward(self,x):
        """
            Autoencodes batch of images x.

            params : 
            x : (B,4,H,W) images
            returns : tuple ((B,4,H,W) prediction , (0,) loss)
        """

        lat_mu, lat_logvar = self.encoder(x)
        lat_vec = self.encoder.sample(lat_mu,lat_logvar)

        out_img = self.decoder(lat_vec)

        assert out_img.shape == x.shape

        loss = self.comp_loss(out_img,x,lat_mu,lat_logvar)
    
        return out_img, loss