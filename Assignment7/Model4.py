# Model 4: Add some depthwise channels to see if it increases the accuracy .

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        self.conv3_depthwise = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
         )

        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
         )

        self.conv4_pointwise = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
             nn.ReLU(),

            )


        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.linearLayer = nn.Linear(16,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3_depthwise(x)
        x = self.conv3_depthwise(x)
        #x = self.conv4_pointwise(x)
        #x = self.conv5_depthwise(x)
        x = self.gap(x)
        x = x.view(-1,16)
        x = self.linearLayer(x)
        return F.log_softmax(x,dim=1)

train_losses = []
test_losses = []
train_acc = []
test_acc = []

# Model1:  Base Model with conv,gap,Linear layer for Classification

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            #nn.BatchNorm2d(64),
            #nn.MaxPool2d(2, 2),
            #nn.Dropout(0.1)
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        #self.conv4 = nn.Sequential(
        #    nn.Conv2d(64, 10, 3, padding=1),
        #    nn.ReLU(),
            #nn.BatchNorm2d(16),
            #nn.MaxPool2d(2, 2),
        #)

        self.linearLayer = nn.Linear(64,10)
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 10, 3, padding=0),
        )

        # self.output_size = self._get_conv_output_size((1, 28, 28))
        #self.fc1 = nn.Sequential(
        #    nn.Linear(self.output_size , 32),
        #)
        self.fc2 = nn.Sequential(
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        #x = self.conv4(x)
        x = x.view(-1,64)
        x = self.linearLayer(x)
        #x = self.fc1(x)
        #x = self.fc2(x)
        return F.log_softmax(x,dim=1)

    def _get_conv_output_size(self, input_size):
        """
        Calculates the output size of the convolutional layers for a given input size.
        This is used to dynamically determine the input size for the first fully connected layer.
        """
        with torch.no_grad():
            # Create a dummy input tensor
            dummy_input = torch.zeros(1, *input_size)
            # Pass the dummy input through the convolutional layers
            output = self.conv4(self.conv3(self.conv2(self.conv1(dummy_input))))
            # Calculate the flattened output size
            output_size = output.numel() // output.size(0)
            return output.shape[1] * output.shape[2] * output.shape[3]


from torchsummary import summary
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))

torch.manual_seed(1)
batch_size = 100


def elastic_transform(image, alpha, sigma):
    """
    Elastic deformation of images as described in [Simard2003].
    """
    #print(type(image))
    image = np.array(image)
    shape = image.shape

    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x = np.clip(x + dx, 0, shape[1] - 1).astype(int)
    y = np.clip(y + dy, 0, shape[0] - 1).astype(int)

    image = image[y, x]
    return image

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                         # transforms.ToPILImage(),
                    # transforms.RandomRotation(3),
                    transforms.RandomAffine(degrees=7, translate=(0.1,0.1), scale=(0.9, 1.1), shear=(-15,15)),
                         transforms.ColorJitter(brightness=0.5, contrast=0.5),
                        transforms.Lambda(lambda x: elastic_transform(x, alpha=7, sigma=7)),

                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)


# Adding a early stopping class
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def l1_regularization(model, lambda1):
    l1_reg = 0.
    for param in model.parameters():
        l1_reg += param.abs().sum()
    return lambda1 * l1_reg

# Define the loss function with L2 regularization
def l2_regularization(model, weight_decay):
    l2_reg = 0.
    for param in model.parameters():
        l2_reg += param.norm(2).square()
    return weight_decay * l2_reg

def loss_fn(output, target, model, l1_lambda, l2_lambda):
    loss = nn.CrossEntropyLoss()(output, target)
    reg_loss = l1_regularization(model, l1_lambda) + l2_regularization(model, l2_lambda)
    return loss + reg_loss

def z_score_outliers(sample, threshold=3):
    mean = torch.mean(sample, dim=0)
    std = torch.std(sample, dim=0)
    z_scores = torch.abs((sample - mean) / std)
    outliers = z_scores > threshold
    return outliers.bool()

# Create an SGD optimizer with exponential decay
def adjust_learning_rate(optimizer, epoch, initial_lr):
    lr = initial_lr * 0.1 ** (epoch // 3)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

  train_loss_sum_epoch = processed - correct
  return train_loss_sum_epoch

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))

model = Net().to(device)
early_stopping = EarlyStopping(patience=2, verbose=True) # patience is number of epochs to wait for improvement

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=4, gamma=0.3)

num_epochs = 15
for epoch in range(1, num_epochs+1):
    print("EPOCH:", epoch)
    val_loss = train(model, device, train_loader, optimizer, epoch)
    # This is to genalize the model faster
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
            print("Early stopping")
            break
    # After each trained epoch , step up
    scheduler.step()
    test(model, device, test_loader)