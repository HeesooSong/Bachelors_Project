# All imports at the topmost part of the code
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import math
import datetime
import shutil
import csv
import argparse

# Definition of all functions


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = math.inf
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    # Loop over the dataset multiple times
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
                # Print current LR
                print("Current LR: {:5f}".format(optimizer.param_groups[0]['lr']))
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    m = nn.Softmax(dim=1)
                    outputs = model(inputs)
                    prob = m(outputs)
                    _, preds = torch.max(prob, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                train_loss_list += [epoch_loss]
                train_acc_list += [epoch_acc]
            else:
                val_loss_list += [epoch_loss]
                val_acc_list += [epoch_acc]

                # Criteria to select which model to save
                #if epoch_loss < best_loss:
                if epoch_acc > best_acc:
                    print(f'New best model found!')
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
        scheduler.step()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f} Best val loss: {:.4f}'.format(best_acc, best_loss))

    if plot:
        plt.title("Training/Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.plot(range(1, num_epochs+1), train_acc_list, label="Train Acc")
        plt.plot(range(1, num_epochs+1), val_acc_list, label="Validation Acc")
        plt.legend()
        plt.savefig("dataset4_3_AL3_Accuracy_Plot.png")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss, best_acc

def test(data):
    correct = 0
    total = 0
    with torch.no_grad():
        with open('dataset4_3_AL3_'+data+'.csv', mode='w') as test_file:
            writer = csv.writer(test_file, delimiter=',')
            for i, (inputs, labels) in enumerate(nonshuffle_dataloaders[data]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model_ft(inputs)
                m = nn.Softmax(dim=1)
                prob = m(outputs.data)
                prob_values, pred_labels = torch.max(prob, 1)
                # loss = criterion(outputs, labels)
                total += labels.size(0)
                correct += (pred_labels == labels).sum().item()
                for x in range(len(inputs)):
                    predicted_prob = prob_values[x].item()
                    if pred_labels[x] == 0:
                        classname = 'neg'
                    else:
                        classname = 'pos'
                    writer.writerow([classname, predicted_prob])
    accuracy = 100*correct/total
    return accuracy

# Now you write your code that utilizes the functions above under __main__


if __name__ == "__main__":

    # Parse experiment name
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-e', '--exp_name', type=str, help="Give the experiment a name")
    #args = parser.parse_args()

    # Ensure an experiment name is provided
    #assert args.exp_name, "Provide experiment name"


    # variables to select which functions to run. Default settings runs only the training code
    training_only = False
    plot = False
    prediction = True
    test1 = False
    test2 = False
    positive_test = False
    test_vid01 = False
    test_vid02 = False
    test_vid03 = False
    test_vid05 = False
    test_vid06 = False
    test_vid07 = False

    batch_size = 32
    num_epochs = 300
    #plt.ion()   # interactive mode : This might not be necessary outside of Jupyter

    # Print out current working directory.
    print("Current working directory: ", os.getcwd())

    print("This code is executed on ", datetime.datetime.now())
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomRotation(5),
            #transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
            #transforms.RandomResizedCrop(224, scale=(0.96, 1.0)),
            #transforms.RandomHorizontalFlip(),
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'unlabeled': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test1': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test2': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'positive_test': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test_vid01': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test_vid02': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test_vid03': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test_vid05': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test_vid06': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test_vid07': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load Data
    #data_dir = "/home/ba4_project/ba4_Hee/Trypanosome/data"
    data_dir = "/home/ba4_project/ba4_Hee/Trypanosome/training_dataset4_3_AL3"
    test_data_dir = "/home/ba4_project/ba4_Hee/timothy_workstation/data/training_dataset1"
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'unlabeled']}
    test_datasets = {x: datasets.ImageFolder(os.path.join(test_data_dir, x),
                                             data_transforms[x])
                     for x in ['test1', 'test2', 'positive_test', 'test_vid01', 'test_vid02', 'test_vid03',
                               'test_vid05', 'test_vid06', 'test_vid07']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    unlabeled_dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                           shuffle=False, num_workers=4)
                            for x in ['unlabeled']}
    nonshuffle_dataloaders = {x: torch.utils.data.DataLoader(test_datasets[x], batch_size=batch_size,
                                                             shuffle=False, num_workers=4)
                              for x in ['test1', 'test2', 'positive_test', 'test_vid01', 'test_vid02', 'test_vid03',
                                        'test_vid05', 'test_vid06', 'test_vid07']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'unlabeled']}
    test_dataset_sizes = {x: len(test_datasets[x]) for x in ['test1', 'test2', 'positive_test']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(class_names)
    print(f'Train image size: {dataset_sizes["train"]}')
    print(f'Validation image size: {dataset_sizes["val"]}')
    print(f'Test image size: {test_dataset_sizes["test1"]}')
    print(f'Test image size: {test_dataset_sizes["test2"]}')
    print(f'Unlabeled image size: {dataset_sizes["unlabeled"]}')
    print(f'Positive test image size: {test_dataset_sizes["positive_test"]}')
    print('-----------------------------------------------------')

    # Train the model
    if training_only:
        # FINETUNING THE CNN
        # Load a model and reset final fully connected layer
        model_ft = torchvision.models.resnet50(pretrained=False)
        #print('model conv: ', model_ft)

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)

        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()

        # Observe that only parameters of final layer are being optimized
        # Stochastic Gradient Decent
        # lr = learning rate = (0.0, 1.0) = controls how quickly the model adapted to the problem
        #      smaller lr require more training epochs 
        # Learning Rate and Step Size prevents overfitting
        optimizer_ft = optim.Adagrad(model_ft.fc.parameters(), lr=0.01, lr_decay=0, weight_decay=0, eps=1e-10)
        # optimizer_ft = optim.Adagrad(model_ft.parameters(), lr=0.01, lr_decay=0.0, weight_decay=0, eps=1e-10)
        #optimizer_ft = optim.Adagrad(model_ft.parameters(), lr=0.1, lr_decay=0.9, weight_decay=2e-4, eps=1e-10)
        #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.8, momentum=0, weight_decay=5e-4)
        #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9)

        # Decay LR by a factor of 0.8 every 5 epochs
        # Sets lr of each parameter group to the initial lr x function
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.8)
        #reduce_on_plateau = lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'max')

        # TRAIN AND EVALUATE
        model_ft, best_val_loss, best_val_acc = train_model(model_ft,
                                                              criterion,
                                                              optimizer_ft,
                                                              exp_lr_scheduler,
                                                              num_epochs = num_epochs)

        torch.save(model_ft, "dataset4_3_AL3_best_model.pth")
    
    if prediction:
        model_ft = torch.load('dataset4_3_AL3_best_model.pth')
        model_ft.eval()
        print("\nPrediction Started")
        filename_list = image_datasets['unlabeled'].imgs

        with torch.no_grad():    
            with open('Prediction4_3_AL3.csv', mode='w') as prediction_file:
                writer = csv.writer(prediction_file, delimiter=',')
                writer.writerow(["Image", "Class", "PredictedProb"])

                for i, (inputs, labels) in enumerate(unlabeled_dataloader['unlabeled']):
                    inputs = inputs.to(device)
                    outputs = model_ft(inputs)
                    m = nn.Softmax(dim=1)
                    prob = m(outputs.data)
                    prob_values, pred_labels = torch.max(prob, 1)
                    
                    for x in range(len(inputs)):
                        predicted_prob = prob_values[x].item()
                        image_num = 32 * i + x      # 32 = batch size
                        filename = filename_list[image_num][0]
                        if pred_labels[x] == 0:
                            classname = 'neg'
                        else:
                            classname = 'pos'

                        writer.writerow([filename, classname, predicted_prob])

    if test1:
        model_ft = torch.load('dataset4_3_AL3_best_model.pth')
        model_ft.eval()
        print("\nTest1 has Started")
        acc = test('test1')
        print('Test1 accuracy: {}%'.format(acc))

    if test2:
        model_ft = torch.load('dataset4_3_AL3_best_model.pth')
        model_ft.eval()
        print("\nTest2 has Started")
        acc = test('test2')
        print('Test2 accuracy: {}%'.format(acc))

    if positive_test:
        model_ft = torch.load('dataset4_3_AL3_best_model.pth')
        model_ft.eval()
        print("\nPositive test has Started")
        acc = test('positive_test')
        print('Positive test accuracy: {}%'.format(acc))

    if test_vid01:
        model_ft = torch.load('dataset4_3_AL3_best_model.pth')
        model_ft.eval()
        print("\nvid01 test has Started")
        acc = test('test_vid01')
        print('vid01 test accuracy: {}%'.format(acc))

    if test_vid02:
        model_ft = torch.load('dataset4_3_AL3_best_model.pth')
        model_ft.eval()
        print("\nvid02 test has Started")
        acc = test('test_vid02')
        print('vid02 test accuracy: {}%'.format(acc))

    if test_vid03:
        model_ft = torch.load('dataset4_3_AL3_best_model.pth')
        model_ft.eval()
        print("\nvid03 test has Started")
        acc = test('test_vid03')
        print('vid03 test accuracy: {}%'.format(acc))

    if test_vid05:
        model_ft = torch.load('dataset4_3_AL3_best_model.pth')
        model_ft.eval()
        print("\nvid05 test has Started")
        acc = test('test_vid05')
        print('vid05 test accuracy: {}%'.format(acc))

    if test_vid06:
        model_ft = torch.load('dataset4_3_AL3_best_model.pth')
        model_ft.eval()
        print("\nvid06 test has Started")
        acc = test('test_vid06')
        print('vid06 test accuracy: {}%'.format(acc))

    if test_vid07:
        model_ft = torch.load('dataset4_3_AL3_best_model.pth')
        model_ft.eval()
        print("\nvid07 test has Started")
        acc = test('test_vid07')
        print('vid07 test accuracy: {}%'.format(acc))
