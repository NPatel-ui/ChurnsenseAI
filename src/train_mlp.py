import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix, classification_report   
from preprocessing import x_train, x_val, y_train, y_val, preprocessor
mlp_model=Pipeline([
    ("preprocessor", preprocessor),
    ("model",MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='adam', max_iter=300,alpha=0.0001,
    learning_rate_init=0.001,early_stopping=True,validation_fraction=0.1,n_iter_no_change=20, random_state=42))
])
mlp_model.fit(x_train,y_train)
y_pred=mlp_model.predict(x_val)
y_prob=mlp_model.predict_proba(x_val)[:,1]
metrics = {
    "Accuracy": accuracy_score(y_val,y_pred),
    "Precision": precision_score(y_val,y_pred,zero_division=0),
    "Recall": recall_score(y_val,y_pred,zero_division=0),
    "F1": f1_score(y_val,y_pred,zero_division=0),
    "Average Precision": average_precision_score(y_val, y_prob),
    "ROC AUC": roc_auc_score(y_val, y_prob)
}
print("Validation Results:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
print("Classification Report:")
print(classification_report(y_val, y_pred,target_names=["No Churn", 'Churn'],zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
network=mlp_model.named_steps['model']
print("NN Details:")
print("Hidden Layers:",network.hidden_layer_sizes)
print("Iterations:",network.n_iter_)
print("Final Training Loss:",round(network.loss_,4))
print("Converged before maximum iterations:",network.n_iter_ < network.max_iter)