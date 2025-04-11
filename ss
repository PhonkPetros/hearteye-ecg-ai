
# === Model 2: XGBoost ===
# === Split training set again to create a validation set for early stopping
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_balanced, y_train_balanced, test_size=0.1, stratify=y_train_balanced, random_state=42
)

# === Model 2: XGBoost with early stopping
xgb_model = xgb.XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_estimators=500,        # Allow many trees, early stopping will decide when to stop
    learning_rate=0.05,      # Slightly slower learning
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist'       # Optional: faster for large data
)

xgb_model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_test, y_test)],
    verbose=True
)


# === Predict and evaluate
xgb_pred = xgb_model.predict(X_test)
print("🔥 XGBoost (Balanced with Early Stopping):\n", classification_report(y_test, xgb_pred, target_names=label_encoder.classes_))

# === Confusion Matrix: XGBoost
cm_xgb = confusion_matrix(y_test, xgb_pred)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=label_encoder.classes_)
disp_xgb.plot(cmap="Oranges", xticks_rotation=45)
plt.title("⚡ XGBoost - Confusion Matrix")
plt.grid(False)
plt.tight_layout()
#plt.show()



from sklearn.naive_bayes import GaussianNB

# === Model 3: Naive Bayes ===
bayes_model = GaussianNB()
bayes_model.fit(X_train_balanced, y_train_balanced)
bayes_pred = bayes_model.predict(X_test)
print("🧪 Naive Bayes (Balanced):\n", classification_report(y_test, bayes_pred, target_names=label_encoder.classes_))

# === Confusion Matrix: Naive Bayes ===
cm_bayes = confusion_matrix(y_test, bayes_pred)
disp_bayes = ConfusionMatrixDisplay(confusion_matrix=cm_bayes, display_labels=label_encoder.classes_)
disp_bayes.plot(cmap="Purples", xticks_rotation=45)
plt.title("🧠 Naive Bayes - Confusion Matrix")
plt.grid(False)
plt.tight_layout()
#plt.show()

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Create the base XGBoost classifier
xgb_clf = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=3,
    random_state=42,
    use_label_encoder=False
)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Run the grid search
grid_search.fit(X_train_balanced, y_train_balanced)

# Best model
best_xgb = grid_search.best_estimator_

# Predict and evaluate
xgb_pred = best_xgb.predict(X_test)
print("🔥 XGBoost (Tuned):\n", classification_report(y_test, xgb_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm_xgb = confusion_matrix(y_test, xgb_pred)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=label_encoder.classes_)
disp_xgb.plot(cmap="Oranges", xticks_rotation=45)
plt.title("⚡ XGBoost - Tuned Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()