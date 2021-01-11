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
        
        # ENCODER
        # Download  pretrained VGG16 model discarding the Linear layers
        self.vgg16 = torchvision.models.vgg16_bn(pretrained=True).features
        
        # Build the DECODER
        self.conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
        self.conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        
        self.conv4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.conv6 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn6 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        
        self.conv7 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn7 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
        self.conv8 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn8 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  
        self.conv9 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn9 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        
        self.conv10 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn10 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv11 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn11 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        
        self.conv12 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn12 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
        self.conv13 = nn.Conv2d(64, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn13 = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            
        
    def forward(self, image):
        
        output = self.vgg16(image)
        
        ind = []
        # Fetch the outpute of mapooling layers in the encoder
        for i, model in enumerate(self.vgg16):
            x = model(x)
            if i in [6, 13, 23, 33, 43]:
                ind.append(x)

        # Decode
        output = F.max_unpool2d(output, ind[-1], kernel_size=2, stride=2)
        output = F.relu(self.bn1(self.conv1(output)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        
        output = F.max_unpool2d(output, ind[-2], kernel_size=2, stride=2)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = F.relu(self.bn6(self.conv6(output))) 

        output = F.max_unpool2d(output, ind[-3], kernel_size=2, stride=2)
        output = F.relu(self.bn7(self.conv7(output)))
        output = F.relu(self.bn8(self.conv8(output)))
        output = F.relu(self.bn9(self.conv9(output)))

        output = F.max_unpool2d(output, ind[-4], kernel_size=2, stride=2)
        output = F.relu(self.bn10(self.conv10(output)))
        output = F.relu(self.bn11(self.conv11(output)))

        output = F.max_unpool2d(output, ind[-5], kernel_size=2, stride=2)
        output = F.relu(self.bn12(self.conv12(output)))
        output = F.relu(self.bn13(self.conv13(output)))

        return nn.Softmax(output, dim=1)
