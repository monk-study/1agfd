import kfp
from kfp import dsl
from kfp.components import create_component_from_func
from typing import Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine

# Component 1: Data extraction from Snowflake
@create_component_from_func
def extract_features_from_snowflake(
    snowflake_account: str,
    snowflake_user: str,
    snowflake_password: str,
    snowflake_database: str,
    snowflake_warehouse: str,
    snowflake_schema: str,
    query: str,
    output_csv_path: str
):
    """Extract features from Snowflake and save to CSV"""
    import pandas as pd
    from snowflake.sqlalchemy import URL
    from sqlalchemy import create_engine
    
    # Create Snowflake connection
    engine = create_engine(URL(
        account=snowflake_account,
        user=snowflake_user,
        password=snowflake_password,
        database=snowflake_database,
        warehouse=snowflake_warehouse,
        schema=snowflake_schema
    ))
    
    # Execute query and save results
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    df.to_csv(output_csv_path, index=False)
    
    return output_csv_path

# Component 2: Data preprocessing
@create_component_from_func
def preprocess_data(
    input_csv_path: str,
    output_features_path: str,
    output_labels_path: str,
    target_column: str,
    categorical_columns: List[str] = None
):
    """Preprocess data and split into features and labels"""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Load data
    df = pd.read_csv(input_csv_path)
    
    # Handle categorical variables
    if categorical_columns:
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
    
    # Split features and labels
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Scale numerical features
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    
    # Save processed data
    X.to_csv(output_features_path, index=False)
    y.to_csv(output_labels_path, index=False)
    
    return output_features_path, output_labels_path

# Component 3: XGBoost Model training
@create_component_from_func
def train_xgboost_model(
    features_path: str,
    labels_path: str,
    model_path: str,
    hyperparameters: Dict
):
    """Train XGBoost model with given hyperparameters"""
    import pandas as pd
    import xgboost as xgb
    import joblib
    
    # Load data
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)
    
    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X, label=y.values.ravel())
    
    # Set default parameters if not provided
    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100
    }
    
    # Update with user-provided parameters
    params = {**default_params, **hyperparameters}
    
    # Train model
    num_rounds = params.pop('n_estimators', 100)
    model = xgb.train(params, dtrain, num_boost_round=num_rounds)
    
    # Save model
    model.save_model(model_path)
    
    return model_path

# Component 4: Model evaluation
@create_component_from_func
def evaluate_xgboost_model(
    model_path: str,
    test_features_path: str,
    test_labels_path: str,
    metrics_path: str
):
    """Evaluate XGBoost model performance"""
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import json
    
    # Load model and test data
    model = xgb.Booster()
    model.load_model(model_path)
    X_test = pd.read_csv(test_features_path)
    y_test = pd.read_csv(test_labels_path)
    
    # Convert to DMatrix
    dtest = xgb.DMatrix(X_test)
    
    # Make predictions
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'auc_roc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Save metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    return metrics_path

# Define the pipeline
@dsl.pipeline(
    name='XGBoost Model Retraining Pipeline',
    description='Pipeline to retrain XGBoost model using features from Snowflake'
)
def xgboost_retraining_pipeline(
    snowflake_account: str,
    snowflake_user: str,
    snowflake_password: str,
    snowflake_database: str,
    snowflake_warehouse: str,
    snowflake_schema: str,
    query: str,
    target_column: str,
    categorical_columns: List[str] = None,
    hyperparameters: Dict = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
):
    # Extract features from Snowflake
    extract_op = extract_features_from_snowflake(
        snowflake_account=snowflake_account,
        snowflake_user=snowflake_user,
        snowflake_password=snowflake_password,
        snowflake_database=snowflake_database,
        snowflake_warehouse=snowflake_warehouse,
        snowflake_schema=snowflake_schema,
        query=query
    )
    
    # Preprocess data
    preprocess_op = preprocess_data(
        input_csv_path=extract_op.output,
        target_column=target_column,
        categorical_columns=categorical_columns
    )
    
    # Train XGBoost model
    train_op = train_xgboost_model(
        features_path=preprocess_op.outputs['output_features_path'],
        labels_path=preprocess_op.outputs['output_labels_path'],
        model_path='model.json',
        hyperparameters=hyperparameters
    )
    
    # Evaluate model
    evaluate_op = evaluate_xgboost_model(
        model_path=train_op.output,
        test_features_path=preprocess_op.outputs['output_features_path'],
        test_labels_path=preprocess_op.outputs['output_labels_path'],
        metrics_path='metrics.json'
    )

# Compile the pipeline
pipeline_func = xgboost_retraining_pipeline
pipeline_filename = 'xgboost_retraining_pipeline.yaml'
kfp.compiler.Compiler().compile(pipeline_func, pipeline_filename)

# Example usage to run the pipeline
client = kfp.Client()
experiment = client.create_experiment('xgboost-retraining')

# Pipeline parameters
run_params = {
    'snowflake_account': 'your_account',
    'snowflake_user': 'your_user',
    'snowflake_password': 'your_password',
    'snowflake_database': 'your_database',
    'snowflake_warehouse': 'your_warehouse',
    'snowflake_schema': 'your_schema',
    'query': 'SELECT * FROM feature_table',
    'target_column': 'target',
    'categorical_columns': ['cat_col1', 'cat_col2'],
    'hyperparameters': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
}

# Submit pipeline run
run = client.run_pipeline(
    experiment.id,
    'XGBoost Retraining Run',
    pipeline_filename,
    params=run_params
)
