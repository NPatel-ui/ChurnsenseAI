
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from preprocessing import (
    x_train,
    x_test,
    y_train,
    y_test,
    preprocessor,
)


# Create the same MLP model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
    )),
])


# Train the model
print("Training MLP model...")
model.fit(x_train, y_train)
# Inspect the trained MLP
mlp = model.named_steps["classifier"]

print("\n===== MLP TRAINING DIAGNOSTICS =====")
print("Iterations completed:", mlp.n_iter_)
print("Final training loss:", round(mlp.loss_, 4))
print("Best validation score:", round(mlp.best_validation_score_, 4))
print("Training loss curve length:", len(mlp.loss_curve_))

# Evaluate training and test data
threshold = 0.45

for dataset_name, X, y in [
    ("TRAINING DATA", x_train, y_train),
    ("TEST DATA", x_test, y_test),
    
]:

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    print(f"\n===== {dataset_name} =====")
    print(f"Accuracy:  {accuracy_score(y, predictions):.4f}")
    print(f"Precision: {precision_score(y, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y, predictions, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y, probabilities):.4f}")
