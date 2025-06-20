# Experiments

| Dataset | Baseline models | CNN-BiLSTM | CNN-BiLSTM on CIC-IDS-2017 | on MAWILab 2011-01 | on 2016-01 | on 2021-01 | RandomForest on 2011-01 | on 2016-01 | on 2021-01 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CIC-IDS-2017 | ✅ | ✅ | | ✅ | | ✅ | | | |
| MAWILab 2011-01 | ✅ | ✅ | ✅ | | ✅ | ✅ | | ✅ | ✅ |
| MAWILab 2016-01 | ✅ | ✅ | | ✅ | | ✅ | ✅ | | ✅ |
| MAWILab 2021-01 | ✅ | ✅ | ✅ | ✅ | ✅ | | ✅ | ✅ | |
| MAWILab 2011-01 + 2016-01 | only RF | ✅ | | | ✅ | ✅ | | | |
| MAWILab 2011-01 + 2021-01| only RF | ✅ | | | ✅ | ✅ | | | |
| MAWILab 2011-01 + 2016-01 + 2021-01| | ✅ | | | ✅ | ✅ | | | |

## Create baselines

- model types
  - RandomForestClassifier
  - DecisionTreeClassifier
  - LogisticRegression
  - XGBClassifier
  - IsolationForest
- run experiments on all datasets with

```
for model_type in RandomForestClassifier DecisionTreeClassifier LogisticRegression XGBClassifier IsolationForest; do \
    uv run python experiments/baseline_cicflowmeter_binary_classifier.py \
        --data-path ~/data/CIC-IDS-2017_GeneratedLabelledFlows.parquet \
        --dataset-name "CIC-IDS-2017" \
        --model-type $model_type \
        -vv; \
    uv run python experiments/baseline_cicflowmeter_binary_classifier.py \
        --data-path ~/data/cicflowmeter_sample_2011-01_n3_000_000.parquet \
        --dataset-name "MAWILab-2011-01-n3_000_000" \
        --model-type $model_type \
        -vv; \
    uv run python experiments/baseline_cicflowmeter_binary_classifier.py \
        --data-path ~/data/cicflowmeter_sample_2016-01_n3_000_000.parquet \
        --dataset-name "MAWILab-2016-01-n3_000_000" \
        --model-type $model_type \
        -vv; \
    uv run python experiments/baseline_cicflowmeter_binary_classifier.py \
        --data-path ~/data/cicflowmeter_sample_2021-01_n3_000_000.parquet \
        --dataset-name "MAWILab-2021-01-n3_000_000" \
        --model-type $model_type \
        -vv; \
    uv run python experiments/baseline_cicflowmeter_binary_classifier.py \
        --data-path ~/data/cicflowmeter_sample_2011-01+2016-01_n6_000_000.parquet \
        --dataset-name "MAWILab-2011-01+2016-01-n6_000_000" \
        --model-type $model_type \
        -vv; \
    uv run python experiments/baseline_cicflowmeter_binary_classifier.py \
        --data-path ~/data/cicflowmeter_sample_2011-01+2021-01_n6_000_000.parquet \
        --dataset-name "MAWILab-2011-01+2021-01-n6_000_000" \
        --model-type $model_type \
        -vv; \
done;
```

## CNN-BiLSTM as binary classifier on CICFlowMeter

### Train with n epochs max

```
epochs=50
```

#### CIC-IDS-2017

```
uv run python experiments/train_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/CIC-IDS-2017_GeneratedLabelledFlows.parquet \
    --dataset-name "CIC-IDS-2017" \
    --epochs $epochs \
    -vv
```

#### MAWILab datasets

```
uv run python experiments/train_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2011-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2011-01-n3_000_000" \
    --epochs $epochs \
    -vv; \
uv run python experiments/train_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2016-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2016-01-n3_000_000" \
    --epochs $epochs \
    -vv; \
uv run python experiments/train_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2021-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2021-01-n3_000_000" \
    --epochs $epochs \
    -vv; \
uv run python experiments/train_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2011-01+2016-01_n6_000_000.parquet \
    --dataset-name "MAWILab-2011-01+2016-01-n6_000_000" \
    --epochs $epochs \
    -vv; \
uv run python experiments/train_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2011-01+2021-01_n6_000_000.parquet \
    --dataset-name "MAWILab-2011-01+2021-01-n6_000_000" \
    --epochs $epochs \
    -vv
```

### Test trained model against different dataset

#### CNN-BiLSTM trained on CIC-IDS-2017

```
mlflow_run_id="9b0c83b43e6d4a5fbbce60e1c0823db8"; \
model_name="CIC-IDS-2017"; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2011-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2011-01-n3_000_000" \
    --model-name $model_name \
    --mlflow-run-id $mlflow_run_id \
    -vv; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2021-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2021-01-n3_000_000" \
    --model-name $model_name \
    --mlflow-run-id $mlflow_run_id \
    -vv
```

#### CNN-BiLSTM trained on MAWILab 2011-01

```
mlflow_run_id="201980eb873f467cb59ea8371b3f1753"; \
model_name="MAWILab-2011-01-n3_000_000"; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/CIC-IDS-2017_GeneratedLabelledFlows.parquet \
    --dataset-name "CIC-IDS-2017" \
    --model-name $model_name \
    --mlflow-run-id $mlflow_run_id \
    -vv; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2016-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2016-01-n3_000_000" \
    --model-name $model_name \
    --mlflow-run-id $mlflow_run_id \
    -vv; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2021-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2021-01-n3_000_000" \
    --model-name $model_name \
    --mlflow-run-id $mlflow_run_id \
    -vv
```

#### CNN-BiLSTM trained on MAWILab 2016-01

```
mlflow_run_id="944db0f555b64b33b84c8016788207d5"; \
model_name="MAWILab-2016-01-n3_000_000"; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2011-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2011-01-n3_000_000" \
    --model-name $model_name \
    --mlflow-run-id $mlflow_run_id \
    -vv; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2021-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2021-01-n3_000_000" \
    --model-name $model_name \
    --mlflow-run-id $mlflow_run_id \
    -vv
```

#### CNN-BiLSTM trained on MAWILab 2021-01

```
mlflow_run_id="2e0c08f7a1944b309cd94eb8669ebb33"; \
model_name="MAWILab-2021-01-n3_000_000"; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/CIC-IDS-2017_GeneratedLabelledFlows.parquet \
    --dataset-name "CIC-IDS-2017" \
    --model-name $model_name \
    --mlflow-run-id $mlflow_run_id \
    -vv; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2011-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2011-01-n3_000_000" \
    --model-name $model_name \
    --mlflow-run-id $mlflow_run_id \
    -vv; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2016-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2016-01-n3_000_000" \
    --model-name $model_name \
    --mlflow-run-id $mlflow_run_id \
    -vv
```

#### RandomForest trained on MAWILab 2011-01

```
mlflow_run_id="d2c8eb407e3442cc86c0c3ad75d76a15"; \
model_name="MAWILab-2011-01-n3_000_000"; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2016-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2016-01-n3_000_000" \
    --model-name $model_name \
    --model-type "RandomForestClassifier" \
    --mlflow-run-id $mlflow_run_id \
    --experiment-name "RandomForest-CICFlowMeter" \
    -vv; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2021-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2021-01-n3_000_000" \
    --model-name $model_name \
    --model-type "RandomForestClassifier" \
    --mlflow-run-id $mlflow_run_id \
    --experiment-name "RandomForest-CICFlowMeter" \
    -vv
```

#### RandomForest trained on MAWILab 2016-01

```
mlflow_run_id="dea1571112a249759febd7dfebbe3db0"; \
model_name="MAWILab-2016-01-n3_000_000"; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2011-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2011-01-n3_000_000" \
    --model-name $model_name \
    --model-type "RandomForestClassifier" \
    --mlflow-run-id $mlflow_run_id \
    --experiment-name "RandomForest-CICFlowMeter" \
    -vv; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2021-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2021-01-n3_000_000" \
    --model-name $model_name \
    --model-type "RandomForestClassifier" \
    --mlflow-run-id $mlflow_run_id \
    --experiment-name "RandomForest-CICFlowMeter" \
    -vv
```

#### RandomForest trained on MAWILab 2021-01

```
mlflow_run_id="49856877a42e49848a5d0a6177da09f6"; \
model_name="MAWILab-2021-01-n3_000_000"; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2011-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2011-01-n3_000_000" \
    --model-name $model_name \
    --model-type "RandomForestClassifier" \
    --mlflow-run-id $mlflow_run_id \
    --experiment-name "RandomForest-CICFlowMeter" \
    -vv; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2016-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2016-01-n3_000_000" \
    --model-name $model_name \
    --model-type "RandomForestClassifier" \
    --mlflow-run-id $mlflow_run_id \
    --experiment-name "RandomForest-CICFlowMeter" \
    -vv
```

#### RandomForest trained on MAWILab 2011-01 + 2016-01

```
mlflow_run_id="2227e55f63554d97bd66953f40625fe1"; \
model_name="MAWILab-2011-01+2016-01-n6_000_000"; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2016-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2016-01-n3_000_000" \
    --model-name $model_name \
    --model-type "RandomForestClassifier" \
    --mlflow-run-id $mlflow_run_id \
    --experiment-name "RandomForest-CICFlowMeter" \
    -vv; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2021-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2021-01-n3_000_000" \
    --model-name $model_name \
    --model-type "RandomForestClassifier" \
    --mlflow-run-id $mlflow_run_id \
    --experiment-name "RandomForest-CICFlowMeter" \
    -vv
```

#### RandomForest trained on MAWILab 2011-01 + 2021-01

```
mlflow_run_id="0f3053d32ad24c5f8ef2c3e320335fa8"; \
model_name="MAWILab-2011-01+2021-01-n6_000_000"; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2016-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2016-01-n3_000_000" \
    --model-name $model_name \
    --model-type "RandomForestClassifier" \
    --mlflow-run-id $mlflow_run_id \
    --experiment-name "RandomForest-CICFlowMeter" \
    -vv; \
uv run python experiments/test_cnn_bilstm_cicflowmeter_binary_classifier.py \
    --data-path ~/data/cicflowmeter_sample_2021-01_n3_000_000.parquet \
    --dataset-name "MAWILab-2021-01-n3_000_000" \
    --model-name $model_name \
    --model-type "RandomForestClassifier" \
    --mlflow-run-id $mlflow_run_id \
    --experiment-name "RandomForest-CICFlowMeter" \
    -vv
```
