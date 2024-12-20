import kfp
from kfp import dsl
from kfp.components import func_to_container_op

def generate_and_preprocess_data():
    """Generate synthetic data and preprocess it for training"""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Create feature data
    features = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(2, 1.5, n_samples),
        'feature3': np.random.uniform(-1, 1, n_samples),
        'categorical1': np.random.choice(['A', 'B', 'C'], n_samples),
        'categorical2': np.random.choice(['X', 'Y', 'Z'], n_samples)
    }
    
    # Create target variable
    target = (features['feature1'] + features['feature2'] > features['feature3']).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(features)
    df['target'] = target
    
    print("Generated data shape:", df.shape)
    
    # Handle categorical variables
    df = pd.get_dummies(df, columns=['categorical1', 'categorical2'])
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['feature1', 'feature2', 'feature3']
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Save processed data
    X_train.to_csv('/tmp/train_features.csv', index=False)
    y_train.to_csv('/tmp/train_labels.csv', index=False)
    X_test.to_csv('/tmp/test_features.csv', index=False)
    y_test.to_csv('/tmp/test_labels.csv', index=False)
    
    return ['/tmp/train_features.csv', '/tmp/train_labels.csv', 
            '/tmp/test_features.csv', '/tmp/test_labels.csv']

def train_xgboost_model(train_features_path: str, train_labels_path: str) -> str:
    """Train XGBoost model with the processed data"""
    import pandas as pd
    import xgboost as xgb
    
    # Load data
    X_train = pd.read_csv(train_features_path)
    y_train = pd.read_csv(train_labels_path)
    
    print("Training data shape:", X_train.shape)
    
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train.values.ravel())
    
    # Set parameters
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'error'],
        'tree_method': 'hist',
        'verbosity': 2
    }
    
    # Train model
    num_rounds = 100
    model = xgb.train(params, dtrain, num_boost_round=num_rounds)
    
    # Save model
    model_path = '/tmp/model.json'
    model.save_model(model_path)
    return model_path

def evaluate_model(model_path: str, test_features_path: str, test_labels_path: str) -> dict:
    """Evaluate the trained model"""
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.metrics import f1_score, roc_auc_score
    
    # Load model and test data
    model = xgb.Booster()
    model.load_model(model_path)
    X_test = pd.read_csv(test_features_path)
    y_test = pd.read_csv(test_labels_path)
    
    print("Test data shape:", X_test.shape)
    
    # Make predictions
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'auc_roc': float(roc_auc_score(y_test, y_pred_proba))
    }
    
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

# Create component operations
generate_op = func_to_container_op(generate_and_preprocess_data, 
                                 base_image='python:3.9',
                                 packages_to_install=['pandas', 'numpy', 'scikit-learn'])

train_op = func_to_container_op(train_xgboost_model, 
                              base_image='python:3.9',
                              packages_to_install=['pandas', 'xgboost'])

evaluate_op = func_to_container_op(evaluate_model, 
                                 base_image='python:3.9',
                                 packages_to_install=['pandas', 'xgboost', 'scikit-learn'])

# Define the pipeline
@dsl.pipeline(
    name='XGBoost Training Pipeline',
    description='A pipeline to train and evaluate XGBoost model on synthetic data'
)
def xgboost_pipeline():
    # Generate and preprocess data
    preprocess_task = generate_op()
    
    # Train model
    train_task = train_op(
        train_features_path=preprocess_task.outputs[0],
        train_labels_path=preprocess_task.outputs[1]
    )
    
    # Evaluate model
    evaluate_task = evaluate_op(
        model_path=train_task.output,
        test_features_path=preprocess_task.outputs[2],
        test_labels_path=preprocess_task.outputs[3]
    )

# Compile the pipeline
kfp.compiler.Compiler().compile(
    pipeline_func=xgboost_pipeline,
    package_path='xgboost_pipeline.yaml'
)
