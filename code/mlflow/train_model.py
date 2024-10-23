from dataloader import *
from imports import *
from helper import *
from optim import *
from models import ECGClassifier


mlflow.enable_system_metrics_logging()
# create dataframe just to log datasets to mlflow
train_ds = pd.read_csv("./data/diagnostics_top4_train_ds.csv")
train_mlflowds = mlflow.data.from_pandas(train_ds, source = LocalArtifactDatasetSource("./data/diagnostics_top4_train_ds.csv"), name="diagnostics_top4_train_ds.csv", targets="Rhythm")
val_ds = pd.read_csv("./data/diagnostics_top4_val_ds.csv")
val_mlflowds = mlflow.data.from_pandas(val_ds, source=LocalArtifactDatasetSource("./data/diagnostics_top4_val_ds.csv"), name="diagnostics_top4_val_ds.csv", targets="Rhythm")
test_ds = pd.read_csv("./data/diagnostics_top4_test_ds.csv")
test_mlflowds = mlflow.data.from_pandas(test_ds, source=LocalArtifactDatasetSource("./data/diagnostics_top4_test_ds.csv"), name="diagnostics_top4_test_ds.csv", targets="Rhythm")

# Load the YAML file
with open('./src/mlflowrunparams.yaml', 'r') as file:
    mlflowparams = yaml.safe_load(file)

perm = {"ecgclassifier" : None, "lead12model": "12LEAD", "transformerecgclassifier": "TRANS",
        "lead12multiinput": "12LEAD" }

train_filename = mlflowparams['mlflowrun']['dataset']['train']
val_filename = mlflowparams['mlflowrun']['dataset']['val']
test_filename = mlflowparams['mlflowrun']['dataset']['test']
batch_size = mlflowparams['mlflowrun']['parameters']['batchsize']
num_epochs = mlflowparams['mlflowrun']['parameters']['epoch']
learningrate = mlflowparams['mlflowrun']['parameters']['learningrate']
experiment_name = mlflowparams['mlflowrun']['experiment']

num_leads = 12
datadir = "./data"
sequence_length = 4096
num_classes = 4 # use only top four classes

# Transformers
d_model = 64
dropout = 0.5

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dataset = ECGChapmanDataset(datadir, train_filename, num_classes=num_classes, seqlen=sequence_length, perm=perm['ecgclassifier'])
test_dataset = ECGChapmanDataset(datadir, test_filename, num_classes=num_classes, seqlen=sequence_length, perm=perm['ecgclassifier'])
val_dataset = ECGChapmanDataset(datadir, val_filename, num_classes=num_classes, seqlen=sequence_length, perm=perm['ecgclassifier'])

# Create a DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# Initialize model, criterion, and optimizer

model = ECGClassifier(sequence_length, num_classes).to(device)
summary(model, input_size=(batch_size, num_leads,sequence_length))

criterion = nn.CrossEntropyLoss()
optimizer =  CustomAdam(model.parameters(), lr=learningrate)

scheduler = None
for input_sample, out_lable, features in train_dataloader:
    print("SAMPLE SHAPE", input_sample.shape)
    print("Feature set shape", features.shape)
    break
signature = infer_signature(input_sample.detach().numpy(), params=mlflowparams['mlflowrun']['parameters'])
# Train the model
mlflow.set_tracking_uri(mlflowparams['mlflowrun']['uri'])
artifact_location = "ftp://mlflowuser:password@ftp_server/"
experiment_id = get_or_create_experiment(experiment_name, artifact_location)
mlflow.set_experiment(experiment_name)

# Start an MLflow run

with mlflow.start_run(run_name=mlflowparams['mlflowrun']['name']) as run:
    mlflow.set_tag("mlflow.note.content", mlflowparams['mlflowrun']['description'])
    #mlflow.set_tag("description", mlflowparams['mlflowrun']['description'])
    mlflow.set_tag("train_dataset", mlflowparams['mlflowrun']['dataset']['train'])
    mlflow.set_tag("val_dataset", mlflowparams['mlflowrun']['dataset']['val'])
    mlflow.log_param("learning_rate", mlflowparams['mlflowrun']['parameters']['learningrate'])
    mlflow.log_param("momentum",  mlflowparams['mlflowrun']['parameters']['momentum'])
    mlflow.log_param("batch_size",  mlflowparams['mlflowrun']['parameters']['batchsize'])
    mlflow.log_param("epoch",  mlflowparams['mlflowrun']['parameters']['epoch'])
    mlflow.log_input(train_mlflowds, context="training", tags={"type": "train"})
    mlflow.log_input(val_mlflowds, context="validation", tags={"type": "val"})
    mlflow.log_input(test_mlflowds, context="validation", tags={"type": "test"})
    #mlflow.log_artifact("/home/vsftpd/mlflowuser")

    #prof.start()
    train_losses, val_losses, train_accuracies, val_accuracies, model = train_and_evaluate(
    model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, device,scheduler=None, profiler=None)
    #prof.stop()

    mlflow.pytorch.log_model(model, mlflowparams['mlflowrun']['name'], signature=signature)

    model.eval()
    val_loss = 0
    predictions = []
    with torch.no_grad():
        for features, labels, add_features in test_dataloader:
            #print("labels", labels)
            #features, labels = add_features.to(device), labels.to(device).argmax(dim=1)
            features, labels = features.to(device), labels.to(device).argmax(dim=1)  # Convert one-hot to class indices
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.flatten().tolist())

    #print("predicted classes", predictions)
    testds = pd.read_csv(os.path.join(datadir, test_filename))
    print(testds.shape, len(predictions), testds.columns)
    testds['prediction'] = predictions
    mapping_dict = {value: key for key, value in classmap.items()}
    testds['predicted_cat'] = testds['prediction'].replace(mapping_dict)

    #print(testds[['Rhythm', 'predicted_cat']].head(4))
    testds.to_csv(os.path.join(datadir, 'diagnostics_top4_predictions.csv'))
    pred_mlflowds = mlflow.data.from_pandas(testds, source="./diagnostics_top4_predictions.csv", name="diagnostics_top4_predictions", targets="Rhythm", predictions="predicted_cat")
    mlflow.log_input(pred_mlflowds, context="predict", tags={"type": "predictions"})
    plot_params = {"TransformerECGClassifier" : "ECGClassifier", "dataset": "raw", "optimizer": 'CustomAdam', "scheduler" : "StepLR"}
    fig_path = plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, **plot_params)
    mlflow.log_artifact(fig_path)
    print("Plot saved to artifacts")

    # create confusion matrix - save this as mlflow artifact
    class_labels = ["SB" , "SR", "AFIB", "ST"]
    cm = confusion_matrix(testds['Rhythm'], testds['predicted_cat'], labels=class_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_labels, yticklabels=class_labels)
    # Adding labels
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    fig.savefig("./result/confusion_matrix.png")
    mlflow.log_artifact("./result/confusion_matrix.png")
