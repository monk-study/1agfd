apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: xgboost-training-
spec:
  entrypoint: model-training-pipeline
  # Define volume for data sharing between steps
  volumes:
    - name: workdir
      emptyDir: {}
    # Secret volume for Snowflake credentials
    - name: snowflake-creds
      secret:
        secretName: snowflake-secret

  # Define templates for workflow steps
  templates:
    - name: model-training-pipeline
      dag:
        tasks:
          - name: extract-data
            template: extract-from-snowflake
          
          - name: preprocess-data
            template: preprocess
            dependencies: [extract-data]
            arguments:
              artifacts:
                - name: raw-data
                  from: "{{tasks.extract-data.outputs.artifacts.raw-data}}"
          
          - name: train-model
            template: train-xgboost
            dependencies: [preprocess-data]
            arguments:
              artifacts:
                - name: processed-features
                  from: "{{tasks.preprocess-data.outputs.artifacts.processed-features}}"
                - name: processed-labels
                  from: "{{tasks.preprocess-data.outputs.artifacts.processed-labels}}"
          
          - name: evaluate-model
            template: evaluate
            dependencies: [train-model]
            arguments:
              artifacts:
                - name: model
                  from: "{{tasks.train-model.outputs.artifacts.model}}"
                - name: test-features
                  from: "{{tasks.preprocess-data.outputs.artifacts.processed-features}}"
                - name: test-labels
                  from: "{{tasks.preprocess-data.outputs.artifacts.processed-labels}}"

    - name: extract-from-snowflake
      container:
        image: python:3.9
        command: [python]
        args:
          - -c
          - |
            import snowflake.connector
            import pandas as pd
            import os
            
            # Read Snowflake credentials from mounted secret
            with open('/snowflake-creds/account') as f:
                account = f.read().strip()
            with open('/snowflake-creds/user') as f:
                user = f.read().strip()
            with open('/snowflake-creds/password') as f:
                password = f.read().strip()
            
            # Connect to Snowflake
            conn = snowflake.connector.connect(
                account=account,
                user=user,
                password=password,
                database='YOUR_DATABASE',
                warehouse='YOUR_WAREHOUSE',
                schema='YOUR_SCHEMA'
            )
            
            # Execute query
            query = """
            SELECT * FROM feature_table
            """
            
            cursor = conn.cursor()
            cursor.execute(query)
            df = cursor.fetch_pandas_all()
            
            # Save data
            df.to_csv('/workdir/raw_data.csv', index=False)
        volumeMounts:
          - name: workdir
            mountPath: /workdir
          - name: snowflake-creds
            mountPath: /snowflake-creds
            readOnly: true
      outputs:
        artifacts:
          - name: raw-data
            path: /workdir/raw_data.csv

    - name: preprocess
      inputs:
        artifacts:
          - name: raw-data
            path: /workdir/raw_data.csv
      container:
        image: python:3.9
        command: [python]
        args:
          - -c
          - |
            import pandas as pd
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            import numpy as np
            
            # Load data
            df = pd.read_csv('/workdir/raw_data.csv')
            
            # Define target and features
            target_col = 'target'  # Update with your target column
            cat_cols = []  # Add your categorical columns
            
            # Handle categorical variables
            for col in cat_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
            
            # Split features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Scale numerical features
            num_cols = X.select_dtypes(include=['int64', 'float64']).columns
            scaler = StandardScaler()
            X[num_cols] = scaler.fit_transform(X[num_cols])
            
            # Save processed data
            X.to_csv('/workdir/processed_features.csv', index=False)
            y.to_csv('/workdir/processed_labels.csv', index=False)
        volumeMounts:
          - name: workdir
            mountPath: /workdir
      outputs:
        artifacts:
          - name: processed-features
            path: /workdir/processed_features.csv
          - name: processed-labels
            path: /workdir/processed_labels.csv

    - name: train-xgboost
      inputs:
        artifacts:
          - name: processed-features
            path: /workdir/processed_features.csv
          - name: processed-labels
            path: /workdir/processed_labels.csv
      container:
        image: python:3.9
        command: [python]
        args:
          - -c
          - |
            import pandas as pd
            import xgboost as xgb
            
            # Load data
            X = pd.read_csv('/workdir/processed_features.csv')
            y = pd.read_csv('/workdir/processed_labels.csv')
            
            # Convert to DMatrix
            dtrain = xgb.DMatrix(X, label=y.values.ravel())
            
            # Set parameters
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss'
            }
            
            # Train model
            num_rounds = 100
            model = xgb.train(params, dtrain, num_boost_round=num_rounds)
            
            # Save model
            model.save_model('/workdir/model.json')
        volumeMounts:
          - name: workdir
            mountPath: /workdir
      outputs:
        artifacts:
          - name: model
            path: /workdir/model.json

    - name: evaluate
      inputs:
        artifacts:
          - name: model
            path: /workdir/model.json
          - name: test-features
            path: /workdir/test_features.csv
          - name: test-labels
            path: /workdir/test_labels.csv
      container:
        image: python:3.9
        command: [python]
        args:
          - -c
          - |
            import pandas as pd
            import xgboost as xgb
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            import json
            
            # Load model and test data
            model = xgb.Booster()
            model.load_model('/workdir/model.json')
            X_test = pd.read_csv('/workdir/test_features.csv')
            y_test = pd.read_csv('/workdir/test_labels.csv')
            
            # Make predictions
            dtest = xgb.DMatrix(X_test)
            y_pred_proba = model.predict(dtest)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted')),
                'recall': float(recall_score(y_test, y_pred, average='weighted')),
                'f1': float(f1_score(y_test, y_pred, average='weighted')),
                'auc_roc': float(roc_auc_score(y_test, y_pred_proba))
            }
            
            # Save metrics
            with open('/workdir/metrics.json', 'w') as f:
                json.dump(metrics, f)
        volumeMounts:
          - name: workdir
            mountPath: /workdir
      outputs:
        artifacts:
          - name: metrics
            path: /workdir/metrics.json

  # Resource requirements and other configurations
  arguments:
    parameters:
      - name: database
        value: "your_database"
      - name: warehouse
        value: "your_warehouse"
      - name: schema
        value: "your_schema"
