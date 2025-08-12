# HW6 Report: Sequence NNets for Machine Translation: RNN, LSTM, Tranformer

## 1. Model Performance Results
### 1.1 Quantitative Metrics
| Model | Final Train Loss | Final Val Loss | BLEU Score | BERT F1 Score |
|-------|-----------------|----------------|------------|---------------|
| RNN   |     5.4138      |   5.8222     |     3.34  | 0.13 |
| Custome RNN |   5.7491   |   6.2771   |  2.45  |   0.01   |
| LSTM  |  3.2654   |  4.5242  |  3.56 |  0.28  |
| Custom LSTM |  3.4757   |  4.5855  |  2.25   |   0.25  |
| Transformer | 1.7135 |  3.03  |  1.59   | 0.56 |

### 1.2 Training Curves
<img width="600" alt="RNN_training_curve" src="https://github.com/user-attachments/assets/2f36b180-99d6-4be7-9cf6-f9c57f46d66b" />
<img width="600" alt="CUSTOM_RNN_training_curve" src="https://github.com/user-attachments/assets/8cdf2535-f988-4736-a330-4eeaa2e87e40" />
<img width="600" alt="LSTM_training_curve" src="https://github.com/user-attachments/assets/607dd30b-3884-4ed9-b8b3-b88d524c26e4" />
<img width="600" alt="CUSTOM_LSTM_training_curve" src="https://github.com/user-attachments/assets/b136d525-2391-4ce3-ad6b-3248ad549255" />
<img width="600" alt="TRANSFORMER_training_curve" src="https://github.com/user-attachments/assets/84c57b0c-c4b6-4fc4-8e43-077acf3cfce8" />

## 2. Model Comparison Analysis
### Compare Different Algorithms
- Transformer learns better (loss), and produces the most semantically accurate translations (BERT F1), but BLEU score is low, possibly due to more varied word choices or paraphrasing.
- LSTM outperforms RNN consistently on all metrics, indicating it better captures sequential dependencies.
- RNN performs worst overall in learning and semantic quality.


