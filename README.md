# Systematic Evaluation of Machine Learning Models for Cytochrome P450 3A4, 2D6, 2C9 Inhibition
## File Structure (Explanation)

### DL: Deep Learning Models
- **GNN: Graph Neural Networks (GIN/GAT/GCN/AttentiveFP)**
  - `data/evalid`: External validation dataset
  - `model_save`: Saved models
  - `data_process_graph.py`: Preprocesses data into graph format and saves it as a `.pt` file, accelerating model training load speed.
  - `model_params.py`: Hyperparameter search range
  - `train.py`: Model training
  - `test.py`: Model testing / external validation

  ##### Usage Flow: `data_process_graph.py -> train.py or test.py`

- **NLP: Natural Language Processing Models (BERT/LSTM/CNN)**
  - `data/evalid`: External validation dataset
  - `pt_model_save`: Saved pre-trained models
  - `model_save`: Saved models
  - `model_params.py`: Hyperparameter search range
  - `seq_encoding.py`: Preprocesses text data into encoded format and saves it as a `.pkl` file, accelerating model training load speed.
  - `train.py`: Model training
  - `test.py`: Model testing / external validation

  ##### Usage Flow: `seq_encoding.py -> train.py or test.py`

### ML: Traditional Machine Learning Models
- `model_save`: Saved models
  - `feature_name_cyp2c9_MORGAN-F_RDKIT.tmp`: Feature names after feature screening
  - `cyp2c9_MORGAN-F_RDKIT_Xgboost.pickle`: Saved model
- `datasets`: Datasets
- `model.py`: Model training
- `test.py`: Model testing / external validation

  ##### Usage Flow: `model.py or test.py`
