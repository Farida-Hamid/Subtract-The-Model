import torch
import torchvision
from torch import nn
import torch.nn.functional as F

class end_to_end(nn.Module):
    """
    Modifies the ResNet model by replacing the 
    last Fully-Connected layer and adding modifications like Dropout.
    
    Args:
        resnet: What version of resnet to use; 18 default.
    """
    def __init__(self):
        super(end_to_end, self).__init__()
        
        # Build the ENCODER
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(64, momentum=0.1, affine=True, track_running_stats=True)
        
        self.enc_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(64, momentum=0.1, affine=True, track_running_stats=True)

        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(128, momentum=0.1, affine=True, track_running_stats=True)
        
        self.enc_conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(128, momentum=0.1, affine=True, track_running_stats=True)

        self.enc_conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc_bn5 = nn.BatchNorm2d(256, momentum=0.1, affine=True, track_running_stats=True)
        
        self.enc_conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.enc_bn6 = nn.BatchNorm2d(256, momentum=0.1, affine=True, track_running_stats=True)
        
        self.enc_conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.enc_bn7 = nn.BatchNorm2d(256, momentum=0.1, affine=True, track_running_stats=True)

        self.enc_conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.enc_bn8 = nn.BatchNorm2d(512, momentum=0.1, affine=True, track_running_stats=True)
        
        self.enc_conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.enc_bn9 = nn.BatchNorm2d(512, momentum=0.1, affine=True, track_running_stats=True)
        
        self.enc_conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.enc_bn10 = nn.BatchNorm2d(512, momentum=0.1, affine=True, track_running_stats=True)

        self.enc_conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.enc_bn11 = nn.BatchNorm2d(512, momentum=0.1, affine=True, track_running_stats=True)
        
        self.enc_conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.enc_bn12 = nn.BatchNorm2d(512, momentum=0.1, affine=True, track_running_stats=True)
        
        self.enc_conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.enc_bn13 = nn.BatchNorm2d(512, momentum=0.1, affine=True, track_running_stats=True)
    
        # Build the DECODER
        self.dec_conv13 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.dec_bn13 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
        self.dec_conv12 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.dec_bn12 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.dec_conv11 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.dec_bn11 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        
        self.dec_conv10 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.dec_bn10 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.dec_conv9 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.dec_bn9 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.dec_conv8 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.dec_bn8 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        
        self.dec_conv7 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.dec_bn7 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
        self.dec_conv6 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.dec_bn6 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  
        self.dec_conv5 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.dec_bn5 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        
        self.dec_conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.dec_bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.dec_conv3 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.dec_bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        
        self.dec_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.dec_bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
        self.dec_conv1 = nn.Conv2d(64, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #self.dec_bn1 = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.softmax = nn.Softmax(dim=1)
            
        
    def forward(self, image):
        
        # Encode
        output = F.relu(self.enc_bn1(self.enc_conv1(image)))
        output = F.relu(self.enc_bn2(self.enc_conv2(output)))
        output, ind1 = F.max_pool2d(output, kernel_size=2, stride=2, return_indices=True)

        output = F.relu(self.enc_bn3(self.enc_conv3(output)))
        output = F.relu(self.enc_bn4(self.enc_conv4(output)))
        output, ind2 = F.max_pool2d(output, kernel_size=2, stride=2, return_indices=True)

        output = F.relu(self.enc_bn5(self.enc_conv5(output)))
        output = F.relu(self.enc_bn6(self.enc_conv6(output)))
        output = F.relu(self.enc_bn7(self.enc_conv7(output)))
        output, ind3 = F.max_pool2d(output, kernel_size=2, stride=2, return_indices=True)

        output = F.relu(self.enc_bn8(self.enc_conv8(output)))
        output = F.relu(self.enc_bn9(self.enc_conv9(output)))
        output = F.relu(self.enc_bn10(self.enc_conv10(output)))
        output, ind4 = F.max_pool2d(output, kernel_size=2, stride=2, return_indices=True)

        output = F.relu(self.enc_bn11(self.enc_conv11(output)))
        output = F.relu(self.enc_bn12(self.enc_conv12(output)))
        output = F.relu(self.enc_bn13(self.enc_conv13(output)))
        output, ind5 = F.max_pool2d(output, kernel_size=2, stride=2, return_indices=True)

        # Decode
        output = F.max_unpool2d(output, ind5, kernel_size=2, stride=2)
        output = F.relu(self.dec_bn13(self.dec_conv13(output)))
        output = F.relu(self.dec_bn12(self.dec_conv12(output)))
        output = F.relu(self.dec_bn11(self.dec_conv11(output)))
        
        output = F.max_unpool2d(output, ind4, kernel_size=2, stride=2)
        output = F.relu(self.dec_bn10(self.dec_conv10(output)))
        output = F.relu(self.dec_bn9(self.dec_conv9(output)))
        output = F.relu(self.dec_bn8(self.dec_conv8(output)))

        output = F.max_unpool2d(output, ind3, kernel_size=2, stride=2)
        output = F.relu(self.dec_bn7(self.dec_conv7(output)))
        output = F.relu(self.dec_bn6(self.dec_conv6(output)))
        output = F.relu(self.dec_bn5(self.dec_conv5(output)))

        output = F.max_unpool2d(output, ind2, kernel_size=2, stride=2)
        output = F.relu(self.dec_bn4(self.dec_conv4(output)))
        output = F.relu(self.dec_bn3(self.dec_conv3(output)))

        output = F.max_unpool2d(output, ind1, kernel_size=2, stride=2)
        output = F.relu(self.dec_bn2(self.dec_conv2(output)))
        #output = F.relu(self.dec_bn1(self.dec_conv1(output)))
        output = self.dec_conv1(output)

        return self.softmax(output)
