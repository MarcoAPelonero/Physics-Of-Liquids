from LoadingUtils.dataset import import_datasets
from LoadingUtils.dataloader import create_dataloaders
from Network.classificationHead import ContrastiveParticleClassifier
from Network.newNetwork import ContrastiveComplexGNN
from Network.trainingFuncs import new_supervised_fine_tuning
import matplotlib.pyplot as plt
import torch

# We want to use a partition that the model has not seen during training, and furthermore
# we want just a few examples to fine tune after unsupervised pre-training.

if __name__ == "__main__":
    train_dataset, validation_dataset, test_dataset = import_datasets(partitions=[0.9, 0.05, 0.05])

    # Cut the valitation in half to use it as a test set

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 8
    hidden_dim = 64
    epochs = 15
    
    train_dataloader = create_dataloaders(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = create_dataloaders(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = create_dataloaders(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    trainedHead = ContrastiveComplexGNN(
        node_dim=5,
        hidden_dim=hidden_dim,
        proj_dim=128,
        k=5,
        num_encoder_layers=2,
        num_message_passes=2
    )

    trainedHead.load_state_dict(torch.load('best_model.pth'))

    untrainedHead = ContrastiveComplexGNN(
        node_dim=5,
        hidden_dim=hidden_dim,
        proj_dim=128,
        k=3,
        num_encoder_layers=2,
        num_message_passes=2,
    )

    trainedClassifier = ContrastiveParticleClassifier(trainedHead, freeze_backbone=True)
    untrainedClassifier = ContrastiveParticleClassifier(untrainedHead, freeze_backbone=False)

    print("\nTrained Classifier Parameters (should be frozen):")
    for name, param in trainedClassifier.named_parameters():
        if 'gnn_backbone' in name:
            print(f"{name}: requires_grad={param.requires_grad} (should be False)")
        else:
            print(f"{name}: requires_grad={param.requires_grad} (should be True)")

    trainedClassifier.to(device)
    untrainedClassifier.to(device)

    data, _ = next(iter(train_dataloader))
    positions = data[:, :, :3].to(device)
    # Pass one batch through 
    for data, labels in val_dataloader:
        data = data.to(device)
        labels = labels.to(device)
        
        # Extract positions (first 3 columns) and q-values (last 2 columns)
        positions = data[..., :3]  # shape: (B, N, 3)
        q_values = data[..., -2:]  # shape: (B, N, 2)
        
        # Get predictions from both classifiers
        trained_predictions = trainedClassifier(
            x=data, 
            positions=positions, 
            q_values=q_values,
            k_spatial=5,  # or whatever you used during training
            k_q=5         # or whatever you used during training
        )
        
        untrained_predictions = untrainedClassifier(
            x=data,
            positions=positions,
            q_values=q_values,
            k_spatial=5,
            k_q=5
        )
        
        print(f"Trained Classifier Predictions shape: {trained_predictions.shape}")
        print(f"Untrained Classifier Predictions shape: {untrained_predictions.shape}")
        break

    class_0_count = 0
    class_1_count = 0

    for labels in train_dataset.labels:
        class_0_count += (labels == 0).sum().item()
        class_1_count += (labels == 1).sum().item()

    
    ratio = class_0_count / class_1_count

    trained_train_loss, trained_train_accuracy, trained_valid_loss, trained_valid_accuracy = new_supervised_fine_tuning(
        model=trainedClassifier,
        train_loader=val_dataloader,
        val_loader=test_dataloader,
        epochs=epochs,
        learning_rate=0.0005,
        ratio=ratio,
        device=device
    )

    untrained_train_loss, untrained_train_accuracy, untrained_valid_loss, untrained_valid_accuracy = new_supervised_fine_tuning(
        model=untrainedClassifier,
        train_loader=val_dataloader,
        val_loader=test_dataloader,
        epochs=epochs,
        learning_rate=0.0005,
        ratio=ratio,
        device=device
    )

    plt.figure(figsize=(12, 5))
    plt.subplot(2, 2, 1)
    plt.plot(trained_train_loss, label='Trained Classifier Loss', color='blue')
    plt.plot(untrained_train_loss, label='Untrained Classifier Loss', color='orange')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(trained_train_accuracy, label='Trained Classifier Accuracy', color='blue')
    plt.plot(untrained_train_accuracy, label='Untrained Classifier Accuracy', color='orange')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(trained_valid_loss, label='Trained Classifier Validation Loss', color='blue')
    plt.plot(untrained_valid_loss, label='Untrained Classifier Validation Loss', color='orange')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(trained_valid_accuracy, label='Trained Classifier Validation Accuracy', color='blue')
    plt.plot(untrained_valid_accuracy, label='Untrained Classifier Validation Accuracy', color='orange')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Take a random batch from the test dataloader
    iterator = iter(test_dataloader)
    test_batch = next(iterator)

    data, labels = test_batch
    positions = data[:, :, :3]

    data = data[0].unsqueeze(0)  # Add batch dimension
    positions = positions[0].unsqueeze(0)  # Add batch dimension

    data = data.to(device)
    positions = positions.to(device)

    # Make sure the predictions are just, so apply a sigmoid and then map 0 and 1
    with torch.no_grad():
        # Extract q_values from the data tensor
        q_values = data[..., -2:]
        q_values = q_values.unsqueeze(0)  
        
        trained_test_predictions = trainedClassifier(
            x=data, 
            positions=positions,
            q_values=q_values,
            k_spatial=5,
            k_q=5
        )
        trained_test_predictions = trained_test_predictions.squeeze(-1)  # Remove last dimension
        trained_test_predictions = torch.sigmoid(trained_test_predictions)
        trained_test_predictions = trained_test_predictions.cpu().detach().numpy()
        trained_test_predictions = (trained_test_predictions > 0.5).astype(int)

        untrained_test_predictions = untrainedClassifier(
            x=data,
            positions=positions,
            q_values=q_values,
            k_spatial=5,
            k_q=5
        )
        untrained_test_predictions = untrained_test_predictions.squeeze(-1)
        untrained_test_predictions = torch.sigmoid(untrained_test_predictions)
        untrained_test_predictions = untrained_test_predictions.cpu().detach().numpy()
        untrained_test_predictions = (untrained_test_predictions > 0.5).astype(int)

    print(f"Trained Test Predictions shape: {trained_test_predictions.shape}")
    print(f"Untrained Test Predictions shape: {untrained_test_predictions.shape}")

    positions = positions[0]
    labels = labels[0].numpy()

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(positions[:, 0].cpu().numpy(), positions[:, 1].cpu().numpy(), positions[:, 2].cpu().numpy(),
                c=labels, cmap='coolwarm', s=5, alpha=0.7)
    ax1.set_title('True Labels')
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(positions[:, 0].cpu().numpy(), positions[:, 1].cpu().numpy(), positions[:, 2].cpu().numpy(),
                c=untrained_test_predictions, cmap='coolwarm', s=5, alpha=0.7)
    ax2.set_title('Untrained Classifier Predictions')
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(positions[:, 0].cpu().numpy(), positions[:, 1].cpu().numpy(), positions[:, 2].cpu().numpy(),
                c=trained_test_predictions, cmap='coolwarm', s=5, alpha=0.7)
    ax3.set_title('Trained Classifier Predictions')
    plt.show()