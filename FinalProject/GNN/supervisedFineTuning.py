from LoadingUtils.dataset import import_datasets
from LoadingUtils.dataloader import create_dataloaders
from Network.classificationHead import ContrastiveParticleClassifier, test_representations
from Network.network import ContrastiveGNN
from Network.trainingFuncs import supervised_fine_tuning
import matplotlib.pyplot as plt
import torch

# We want to use a partition that the model has not seen during training, and furthermore
# we want just a few examples to fine tune after unsupervised pre-training.

if __name__ == "__main__":
    train_dataset, validation_dataset, test_dataset = import_datasets()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 8
    hidden_dim = 64
    epochs = 1
    
    train_dataloader = create_dataloaders(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = create_dataloaders(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = create_dataloaders(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    trainedHead = ContrastiveGNN(
        node_dim=5,
        hidden_dim=hidden_dim,
        proj_dim=128,
        k=3,
        num_encoder_layers=2,
        num_message_passes=2
    )

    trainedHead.load_state_dict(torch.load('best_model.pth'))

    untrainedHead = ContrastiveGNN(
        node_dim=5,
        hidden_dim=hidden_dim,
        proj_dim=128,
        k=3,
        num_encoder_layers=2,
        num_message_passes=2,
    )

    pretrained_acc = test_representations(trainedHead, val_dataloader, device)

    untrained_acc = test_representations(untrainedHead, val_dataloader, device)

    print(f"Pretrained linear probe accuracy: {pretrained_acc:.4f}")
    print(f"Untrained linear probe accuracy: {untrained_acc:.4f}")

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

    print("k=3 edges:", trainedHead.get_edges(positions, k=3).shape)  # Should be ~3×num_nodes
    print("k=15 edges:", trainedHead.get_edges(positions, k=15).shape) 
    # Pass one batch through 
    for data, labels in val_dataloader:
        data = data.to(device)
        labels = labels.to(device)

        positions = data[:, :, :3]

        posittions = positions.to(device)

        trained_predictions = trainedClassifier(data, positions)
        untrained_predictions = untrainedClassifier(data, positions)

        print(f"Trained Classifier Predictions: {trained_predictions.shape}")
        print(f"Untrained Classifier Predictions: {untrained_predictions.shape}")
        break

    class_0_count = 0
    class_1_count = 0

    for labels in train_dataset.labels:
        class_0_count += (labels == 0).sum().item()
        class_1_count += (labels == 1).sum().item()

    
    ratio = class_0_count / class_1_count

    trained_train_loss, trained_train_accuracy, trained_valid_loss, trained_valid_accuracy = supervised_fine_tuning(
        model=trainedClassifier,
        train_loader=val_dataloader,
        val_loader=test_dataloader,
        epochs=epochs,
        learning_rate=0.0005,
        ratio=ratio,
        device=device
    )

    untrained_train_loss, untrained_train_accuracy, untrained_valid_loss, untrained_valid_accuracy = supervised_fine_tuning(
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
        trained_test_predictions = trainedClassifier(data,  positions)
        trained_test_predictions = trained_test_predictions.squeeze(-1)  # Remove last dimension
        trained_test_predictions = torch.sigmoid(trained_test_predictions)
        trained_test_predictions = trained_test_predictions.cpu().detach().numpy()
        trained_test_predictions = (trained_test_predictions > 0.5).astype(int)

        untrained_test_predictions = untrainedClassifier(data, positions)
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