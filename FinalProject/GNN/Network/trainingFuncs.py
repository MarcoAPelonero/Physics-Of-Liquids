import torch
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss

def save_model(model, save_path):
    """
    Save the model state dictionary to the specified path.

    Args:
        model: The GNN model to save.
        save_path (str): Path to save the model.
    """
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def train_epoch(model, data_loader, loss_fn, metric_fn, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: The GNN model to train.
        data_loader: DataLoader for the training set.
        loss_fn: Loss function to compute the loss.
        metric_fn: Metric function to compute the metric.
        optimizer: Optimizer for updating model parameters.
        device (str): Device to run the training on ('cpu' or 'cuda').

    Returns:
        tuple: Average loss and metric for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_metric = 0.0
    num_batches = len(data_loader)

    for data, _ in data_loader:
        optimizer.zero_grad()
        
        positions = data[:, :, :3]
        
        data = data.to(device)
        positions = positions.to(device)

        z1, _ = model(data, positions, 3)
        z2, _ = model(data, positions, 15)

        loss = loss_fn(z1, z2)
        metric = metric_fn(z1, z2)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_metric += metric

    avg_loss = total_loss / num_batches
    avg_metric = total_metric / num_batches

    return avg_loss, avg_metric

def validate_epoch(model, data_loader, loss_fn, metric_fn, device):
    """
    Validate the model for one epoch.

    Args:
        model: The GNN model to validate.
        data_loader: DataLoader for the validation set.
        loss_fn: Loss function to compute the loss.
        metric_fn: Metric function to compute the metric.
        device (str): Device to run the validation on ('cpu' or 'cuda').

    Returns:
        tuple: Average loss and metric for the epoch.
    """
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    num_batches = len(data_loader)

    with torch.no_grad():
        for data, _ in data_loader:
            positions = data[:, :, :3]

            data = data.to(device)
            positions = positions.to(device)

            z1, _ = model(data, positions, 3)
            z2, _ = model(data, positions, 15)

            loss = loss_fn(z1, z2)
            metric = metric_fn(z1, z2)

            total_loss += loss.item()
            total_metric += metric

    avg_loss = total_loss / num_batches
    avg_metric = total_metric / num_batches

    return avg_loss, avg_metric

def train(model, train_loader, val_loader, 
          epochs, learning_rate, weight_decay,
          loss_fn, metric_fn, 
          device = 'cpu', patience=10, save_path=None):
    """
    Train the GNN model with early stopping and save the best model.

    Args:
        model: The GNN model to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        device (str): Device to run the training on ('cpu' or 'cuda').
        patience (int): Number of epochs to wait for improvement before stopping.
        save_path (str): Path to save the best model.

    Returns:
        tuple: Training loss and metric history, validation loss and metric history.
    """
    
    train_losses = []
    train_metrics = []
    val_losses = []
    val_metrics = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.to(device)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in tqdm(range(epochs), total=epochs, desc="Training Progress"):
        model.train()
        train_loss = 0.0
        train_metric = 0.0
        val_loss = 0.0
        val_metric = 0.0

        train_loss, train_metric = train_epoch(model, train_loader, loss_fn, metric_fn, optimizer, device)
        train_losses.append(train_loss)
        train_metrics.append(train_metric)

        val_loss, val_metric = validate_epoch(model, val_loader, loss_fn, metric_fn, device)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if save_path:
                save_model(model, save_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement.")
            break
    print("Training complete.")
    return train_losses, train_metrics, val_losses, val_metrics

def supervised_fine_tuning(model, train_loader, val_loader, epochs, learning_rate, ratio, device='cpu'):
    train_losses = []
    train_metrics = []

    val_losses = []
    val_metrics = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    
    pos_weight = torch.tensor([ratio], device=device)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
    
    for epoch in tqdm(range(epochs), total=epochs, desc="Supervised Fine-Tuning Progress"):

        epoch_loss = 0.0
        correct = 0
        total = 0

        model.train()

        for data, label in train_loader:
            
            positions = data[:, :, :3]
            positions = positions.to(device)
            data = data.to(device)
            labels = label.to(device).float()

            optimizer.zero_grad()
            
            outputs = model(data, positions)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            outputs = outputs.squeeze(-1)  # Remove last dimension
            outputs = torch.sigmoid(outputs)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        
        train_losses.append(avg_loss)
        train_metrics.append(accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, label in val_loader:
                positions = data[:, :, :3]
                positions = positions.to(device)
                data = data.to(device)
                labels = label.to(device).float()

                outputs = model(data, positions)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                outputs = outputs.squeeze(-1)  # Remove last dimension
                outputs = torch.sigmoid(outputs)
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total

        val_losses.append(avg_val_loss)
        val_metrics.append(accuracy)

        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.4f}")
    
    return train_losses, train_metrics, val_losses, val_metrics 