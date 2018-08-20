# By: Ryan McVicker
# Date started: 06/01/18
# Date completed: xx/xx/xx
# Project: AI Programming with Python nanodegree

## IMPORT
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
from collections import OrderedDict
import argparse

## def MAIN
def main():
    # Parse Arguments
    args = get_args()
    
    # Set Directories
    data_dir = args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    save_dir = args.save_dir

    # Define Transforms
    train_transforms, valid_transforms, test_transforms = define_transforms()
    
    # Load Datasets
    train_data, valid_data, test_data = load_datasets(train_transforms, valid_transforms, test_transforms, train_dir, valid_dir, test_dir)
    
    # Define Dataloaders
    trainloader, validloader, testloader = define_dataloaders(train_data, valid_data, test_data)

    # Set Variables
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu
    steps = 0
    running_loss = 0
    print_every = 30
    criterion = nn.NLLLoss()
    
    # Select Model
    model = select_model(arch, hidden_units)
    
    # Set optimizer after model.classifier has been set
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # GPU or CPU
    if gpu and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    
    # Train and Validate Model
    train_valid_model(model, epochs, trainloader, gpu, optimizer, criterion, steps, running_loss, print_every, validloader)
    
    # Store Checkpoint
    model.class_to_idx = train_data.class_to_idx
    store_checkpoint(arch, model, optimizer, epochs, save_dir, learning_rate)

## def GET_ARGS
def get_args():
    parser = argparse.ArgumentParser(description='parse user input')
    parser.add_argument('data_directory', type=str, help='main image folder name')
    parser.add_argument('--save_dir', type=str, help='user defined directory to save checkpoints, default is /save', default='/save')
    parser.add_argument('--arch', type=str, help='choose pre-trained model: vgg16 or densenet121, default is VGG16 if none is chosen', default='vgg16')
    parser.add_argument('--learning_rate', type=float, help='choose the learning rate, between 0.0001 and 1.0, default is 0.01', default='0.01')
    parser.add_argument('--hidden_units', type=int, help='choose number of hidden nodes, between 102 and 10000, as an integer, default is 512', default='512', choices=range(102, 10000))
    parser.add_argument('--epochs', type=int, help='choose number of epochs, between 1 and 100, as an integer, default is 20', default='20', choices=range(1, 100))
    parser.add_argument('--gpu', type=bool, help='choose to use GPU, between True or False, default is True', default='True')
    
    args = parser.parse_args()
    
    return args

## def DEFINE_TRANSFORMS
def define_transforms():
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    return train_transforms, valid_transforms, test_transforms

## def LOAD_DATASETS
def load_datasets(train_transforms, valid_transforms, test_transforms, train_dir, valid_dir, test_dir):
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    return train_data, valid_data, test_data

## def DEFINE_DATALOADERS
def define_dataloaders(train_data, valid_data, test_data):
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return trainloader, validloader, testloader

## def SELECT_MODEL
def select_model(arch, hidden_units):
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Set Dropout Rate
    drop_rate = 0.5
    
    # Build Classifier
    if arch == 'densenet121':
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, hidden_units)),
            ('relu1', nn.ReLU(inplace = True)),
            ('dropout1', nn.Dropout(p=drop_rate)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    else:
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 4096)),
            ('relu1', nn.ReLU(inplace = True)),
            ('dropout1', nn.Dropout(p=drop_rate)),
            ('fc2', nn.Linear(4096, hidden_units)),
            ('relu2', nn.ReLU(inplace = True)),
            ('dropout2', nn.Dropout(p=drop_rate)),
            ('fc3', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    
    model.classifier = classifier
    
    return model

## def TRAIN_VALID_MODEL
def train_valid_model(model, epochs, trainloader, gpu, optimizer, criterion, steps, running_loss, print_every, validloader):
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        
        for images, labels in iter(trainloader):
            # Wrap images and labels in Variables so we can calculate gradients
            images = Variable(images)
            labels = Variable(labels)
            
            # If using GPU load images and labels to GPU
            if gpu and torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            
            # Clear the gradients from all Variables
            optimizer.zero_grad()
            
            # Run FeedFordward on training dataset, resulting in outputs
            output = model.forward(images)

            # Calculate losses, diff of outputs and labels
            loss = criterion(output, labels)

            # Run Backpropogation from losses
            loss.backward()

            # Update weights from backpropogation
            optimizer.step()
            
            # Track steps to determine when to run evaluation
            steps += 1

            # Track training loss to print info during evaluation
            running_loss += loss.data[0]
                
            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                accuracy = 0
                valid_loss = 0
                
                # iterate thru validation dataset
                for ii, (images, labels) in enumerate(validloader):
                    # If using GPU load images and labels to GPU
                    if gpu and torch.cuda.is_available():
                        images = images.cuda()
                        labels = labels.cuda()
                
                    # Volatile no longer working on PyTorch 0.4, Wrap images and labels in Variables so we can calculate gradients
                    images = Variable(images)
                    labels = Variable(labels)
                
                    # Run FeedFordward on validation dataset, resulting in outputs
                    output = model.forward(images)

                    # Calculate validation losses, diff of outputs and labels
                    valid_loss += criterion(output, labels).data[0]
                
                    # Model's output is log-softmax, take exp to get the prob's
                    ps = torch.exp(output).data
                
                    # Class with highest probability is our predicted class,
                    # compare with true label, results in boolean integer
                    equality = (labels.data == ps.max(1)[1])

                    # Accuracy is number of correct predictions divided by all
                    # predictions, just take the mean, first float boolean integer
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
                
                print("Train Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                
                running_loss = 0

                # Make sure dropout is on for training
                model.train()

## def STORE_CHECKPOINT
def store_checkpoint(arch, model, optimizer, epochs, save_dir, learning_rate):
    checkpoint = {'arch': arch,
                  'model': model,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx(),
                  'learning_rate': learning_rate
                 }
    
    filepath = save_dir + '/MCVICKER-' + int(round(time.time())) + '.pth'
    
    torch.save(checkpoint, filepath)

if __name__ == '__main__':
    main()