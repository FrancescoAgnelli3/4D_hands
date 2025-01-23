import torch.nn as nn
import controldiffeq
import torch

class BasicCNN(nn.Module):
    def __init__(self, input_channels, output_channels, hid_channels, kernel_size, stride=1, padding=0):
        """
        Initialize the CNN.

        Args:
            input_channels (int): Number of input channels (C in BxTxNxC).
            output_channels (int): Number of output channels (C_out in BxC_out).
            hid_channels (int): Number of filters for the convolutional layers.
            kernel_size (int or tuple): Kernel size for the convolutional layers.
            stride (int or tuple, optional): Stride for the convolutional layers. Default is 1.
            padding (int or tuple, optional): Padding for the convolutional layers. Default is 0.
        """
        super(BasicCNN, self).__init__()

        # Define layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, 
                               out_channels=hid_channels, 
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=hid_channels, 
                               out_channels=hid_channels*2, 
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding=padding)
        self.conv3 = nn.Conv2d(in_channels=hid_channels*2, 
                               out_channels=hid_channels*2, 
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding=padding)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2*hid_channels, output_channels)

    def forward(self, spline, times):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape BxTxNxC.

        Returns:
            torch.Tensor: Output tensor of shape BxC_out.
        """
        x = []
        for i in range(len(times)):
            x.append(spline.evaluate(i))
        x = torch.stack(x, dim=0).permute(1, 0, 2, 3)

        # Reshape input to BxCxTxN
        x = x.permute(0, 3, 1, 2)  # BxTxNxC -> BxCxTxN

        # Apply first convolution and activation
        x = self.conv1(x)
        x = self.relu(x)

        # Apply second convolution and activation
        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        # Global average pooling
        x = self.global_pool(x)  # BxCx1x1

        # Flatten and apply fully connected layer
        x = x.view(x.size(0), -1)  # BxC
        x = self.fc(x)  # BxC_out

        return x