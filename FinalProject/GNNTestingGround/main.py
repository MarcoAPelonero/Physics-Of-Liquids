from Network.geomArch import AmorphousParticleGNN
from Training.dataImport import create_data_loaders
from Training.utils import train
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Configuration
    config = {
        'hidden_dim':128,
        'num_layers': 4,
        'proj_dim': 128,
        'box_size': 1.0,
        'k_list': [3, 15],       # Local and global k-values
        'temperature': 0.1,      # NT-Xent temperature
        'num_epochs': 40,
        'batch_size': 24,        # Should match your dataloader
        'lr': 0.001,
        'save_path': "best_amorphous_model.pth"
    }
    
    # Initialize model
    model = AmorphousParticleGNN(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        proj_dim=config['proj_dim'],
        box_size=config['box_size']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("Starting training...")
    print(f"Using device: {next(model.parameters()).device}")
    print(f"Batch size: {config['batch_size']}, Particles per batch: {config['batch_size'] * 900}")
    print(f"Contrastive views: k={config['k_list']}")
    
    data_directory = r"C:\Users\marcu\Desktop\github repos\Physics-Of-Liquids\FinalProject\ProjectGNN\Dataset\hard_sphere_ensemble"
    train_loader, val_loader, test_loader = create_data_loaders(data_directory, corrupted=True)
    # Run training
    trained_model, train_losses, val_losses, train_metric, vali_metric, lr_history = train(
        model,
        train_loader,
        val_loader,
        k_list=config['k_list'],
        temperature=config['temperature'],
        num_epochs=config['num_epochs'],
        lr=config['lr'],
        save_path=config['save_path'],  # <-- Comma was missing here
        patience=10,
        min_lr=1e-6,
        factor=0.5,
        grad_clip=1.0,
        use_amp=False
    )
    
    print("Training completed!")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")

    # Plot losses and metric on 4x4 subplot matrix
    fig, axs = plt.subplots(2, figsize=(8,4))
    axs = axs.flatten()
    axs[0].plot(train_losses, label='Train Loss', color='blue')
    axs[0].plot(val_losses, label='Validation Loss', color='orange')
    axs[0].set_title('Losses')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[1].plot(train_metric, label='Train Metric', color='green')
    axs[1].plot(vali_metric, label='Validation Metric', color='red')
    axs[1].set_title('Metrics')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Metric')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("training_results.png")
    print("Saved training results to 'training_results.png'")
    plt.show()
    # plot and save lr history
    plt.figure(figsize=(8, 4))
    plt.plot(lr_history, label='Learning Rate', color='purple')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_rate_schedule.png")
    print("Saved learning rate schedule to 'learning_rate_schedule.png'")
    plt.show()

    # Save all the data
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metric': train_metric,
        'vali_metric': vali_metric,
        'lr_history': lr_history
    }, "training_data.pth")
    # Save final model
    torch.save(trained_model.state_dict(), "final_model.pth")
    print("Saved final model to 'final_model.pth'")