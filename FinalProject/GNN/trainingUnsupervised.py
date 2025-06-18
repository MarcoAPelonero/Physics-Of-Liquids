from LoadingUtils.dataset import import_datasets
from LoadingUtils.dataloader import create_dataloaders, visualize_features
from Network.network import ContrastiveGNN
from Network.trainingFuncs import train
from Network.lossMetric import nt_xent_loss_2n_3d, nt_xent_accuracy_2n_3d
import matplotlib.pyplot as plt
import torch

VISUALIZE_FEATURES = False  # Set to True to visualize features before training

if __name__ == "__main__":
    training_dataset, validation_dataset, test_dataset = import_datasets()

    batch_size = 8
    hidden_dim = 64

    train_dataloader = create_dataloaders(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = create_dataloaders(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = create_dataloaders(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Print shape of the first batch in each DataLoader
    for data, labels in train_dataloader:
        print(f"Train DataLoader Batch - Data shape: {data.shape}, Labels shape: {labels.shape}")
        break

    print(f"Train DataLoader: {len(train_dataloader)} batches")
    print(f"Validation DataLoader: {len(val_dataloader)} batches")
    print(f"Test DataLoader: {len(test_dataloader)} batches")

    model = ContrastiveGNN(
        node_dim=5,
        hidden_dim=hidden_dim,
        proj_dim=128,
        k=5,
        num_encoder_layers=2,
        num_message_passes=2,
        use_attention=False,
        use_skip_connections=True,
        use_batch_norm=False,
    )

    # Display features space pre training with t-SNE
    if VISUALIZE_FEATURES:
        visualize_features(
            model=model,
            dataloader=test_dataloader,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    train_loss, train_metric, val_loss, val_metric = train(
        model = model,
        train_loader = train_dataloader,
        val_loader = val_dataloader,
        epochs= 8,
        learning_rate= 0.001,
        weight_decay= 1e-5,
        loss_fn= nt_xent_loss_2n_3d,
        metric_fn= nt_xent_accuracy_2n_3d,
        device= 'cuda' if torch.cuda.is_available() else 'cpu',
        patience= 1,
        save_path= 'best_model.pth' 
    )

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_metric, label='Train Metric')
    plt.plot(val_metric, label='Validation Metric')
    plt.title('Metric over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.legend()
    plt.tight_layout()
    plt.show()

    if VISUALIZE_FEATURES:
        visualize_features(
            model=model,
            dataloader=test_dataloader,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )