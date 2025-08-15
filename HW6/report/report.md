# HW6 Report: Sequence NNets for Machine Translation: RNN, LSTM, Tranformer

## 1. Model Performance Results
### 1.1 Quantitative Metrics
| Model | Final Train Loss | Final Val Loss | BLEU Score | BERT F1 Score |
|-------|-----------------|----------------|------------|---------------|
| RNN   |     5.6046     |   6.1092  |     2.67  | 0.09 |
| Custome RNN |   5.7518  |   6.2461  | 1.85 |   0.015   |
| LSTM  |  3.1608   |  4.4750 |  6.06 |  0.29  |
| Custom LSTM |  3.1608  |  4.4750  |  6.06  |   0.29 |
| Transformer | 1.6903 |  2.1834  |  1.33   | 0.55 |

### 1.2 Training Curves
<img width="600" alt="RNN_training_curve" src="https://github.com/user-attachments/assets/393d0539-e8b2-45a4-9af7-427bb8691988" />
<img width="600" alt="CUSTOM_RNN_training_curve" src="https://github.com/user-attachments/assets/0edf57f7-10e9-4eb1-aadc-2125df43bff8" />
<img width="600" alt="LSTM_training_curve" src="https://github.com/user-attachments/assets/f19fb658-d590-47e2-a0a5-7568ba2c3497" />
<img width="600" alt="CUSTOM_LSTM_training_curve" src="https://github.com/user-attachments/assets/f5b29969-c8a6-4bad-a2c3-8c208356cf3b" />
<img width="600" alt="TRANSFORMER_training_curve" src="https://github.com/user-attachments/assets/be940ec4-5f89-4582-b8ca-103ace09bd5d" />


## 2. Model Comparison Analysis
- RNN → LSTM → Transformer shows clear progression
- Semantic understanding: 0.09 → 0.29 → 0.55 (6x improvement)

## 3. Challenges Faced
### 3.1 **Custom RNN Failure**
- **Issue:** BERT F1 = 0.015 (near zero), indicating complete semantic failure

### 3.2 **Transformer BLEU Paradox**
- **Issue:** Highest BERT (0.55) but lowest BLEU (1.33)
- **Insight:** Transformer generates paraphrases (semantically correct) rather than literal translations

