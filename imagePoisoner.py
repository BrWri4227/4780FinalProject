
#MODIFIED CODE
def fgsmAttack(image, epsilon, data_grad): #FGSM generator
    sign_data_grad = data_grad.sign() #Get the sign of the data gradient
    attackIMG = image + epsilon * sign_data_grad #Pertubate
    attackIMG = torch.clamp(attackIMG, 0, 1) 
    return attackIMG

def fgsmTester(test_loader, net, criterion, optimizer, epoch, device, epsilon): #FGSM Tester
    global best_prec, writer

    net.eval()

    test_loss = 0
    correct = 0
    total = 0
    totalConfidence = 0
    logger.info(" === FGSM ===")

    for batch_index, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        
        inputs.requires_grad = True
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        net.zero_grad()  # Zero the existing gradients
        loss.backward()  # Backward pass
        # Collect the gradient of the input with respect to the loss
        data_grad = inputs.grad.data
        # Call FGSM Attack
        attackImages = fgsmAttack(inputs, epsilon, data_grad)

        # Forward pass with perturbed inputs
        outputs = net(attackImages)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = outputs.max(1) #get the index of the max log-probability
        total += targets.size(0) #calculate total number of images
        correct += predicted.eq(targets).sum().item() #calculate total number of correct predictions
        probs = torch.nn.functional.softmax(outputs, dim=1) #calculate confidence
        # totalConfidence += torch.max(probs) 
        totalConfidence += torch.max(probs, dim=1)[0].sum().item() #calculate total confidence
    logger.info(
        "   == FGSM test loss: {:.3f} | FGSM test acc: {:6.3f}% | FGSM test Average confidence: {:.3f}%".format(
            test_loss / (batch_index + 1), 100.0 * correct / total, 100.0 * (totalConfidence / total)
        )
    )