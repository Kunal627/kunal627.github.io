from imports import *

def get_or_create_experiment(experiment_name, artifact_location):
    # Check if the experiment already exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment:
        # If it exists, return its ID
        print(f"Experiment '{experiment_name}' already exists with ID: {experiment.experiment_id}")
        return experiment.experiment_id
    else:
        # If it does not exist, create a new one
        experiment_id = mlflow.create_experiment(experiment_name, artifact_location)
        print(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
        return experiment_id

def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

def evaluate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()  # Convert probabilities to binary

            # Calculate number of correct predictions
            correct += (predicted == labels).sum().item()
            total += labels.size(0) * labels.size(1)  # Total elements

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    print(f'accuracy, {accuracy:.4f},   loss, {avg_loss:.4f}')

    return avg_loss, accuracy

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, **kwargs):
    epochs = range(1, len(train_losses) + 1)
    filename = kwargs.get("modelclass","metrics-graph") + "-" \
               + kwargs.get("dataset","dataset")   + "-" \
               + kwargs.get("optimizer", "optim") + "-" \
               + kwargs.get("scheduler", "schd") \
               + "-metrics-plot.png"
    outpath = os.path.join("./result", filename)

    plt.figure(figsize=(14, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    #plt.show()
    plt.savefig(outpath)
    return outpath

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler=None, profiler=None):
    model = model.to(device)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        if profiler is not None:
            profiler.step()
        epoch_start_time = time.time()
        # Training phase
        print("training started for epoch", epoch+1)
        model.train()
        train_loss = 0
        correct_predictions = 0
        total_samples = 0
        for features, labels, add_features in train_loader:
            features, labels = features.to(device), labels.to(device).argmax(dim=1)  # Convert one-hot to class indices
            #features, labels = add_features.to(device), labels.to(device).argmax(dim=1)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        train_losses.append(train_loss / total_samples)
        train_accuracies.append(correct_predictions / total_samples)
        # Evaluation phase
        print("starting evaluation!!! epoch ", epoch+1)
        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for features, labels, add_features in val_loader:
                features, labels = features.to(device), labels.to(device).argmax(dim=1)  # Convert one-hot to class indices
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        val_losses.append(val_loss / total_samples)
        val_accuracies.append(correct_predictions / total_samples)
        epoch_end_time = time.time()  # End timing the epoch
        epoch_duration = epoch_end_time - epoch_start_time  # Calculate duration

        mlflow.log_metric("val_loss", val_losses[-1], step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracies[-1], step=epoch)
        mlflow.log_metric("train_loss", train_losses[-1], step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracies[-1], step=epoch)
        
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, '
              f'Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}, '
              f'Time: {epoch_duration:.2f} seconds')
        
    return train_losses, val_losses, train_accuracies, val_accuracies, model

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("./logs/trace_" + str(p.step_num) + ".json")
