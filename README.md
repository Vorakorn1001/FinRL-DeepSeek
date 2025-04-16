# FinRL: Adaptive Model Selection for Reinforcement Learning in Stock Trading

This repository extends the FinRL-DeepSeek project, integrating adaptive model selection into risk-sensitive reinforcement learning for trading agents.

---

## 📄 Overview

FinRL: Adaptive Model Selection enhances the FinRL-DeepSeek framework by enabling dynamic selection between multiple reinforcement learning agents based on market conditions. This approach aims to improve trading performance by switching between:

- **PPO**: Standard Proximal Policy Optimization
- **CPPO-DeepSeek**: Risk-aware Conditional PPO with LLM guidance

---

## 🚀 Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/Vorakorn1001/FinRL-DeepSeek.git
   cd FinRL-DeepSeek
   ```

2. **Run the installer**

   ```bash
    installation_script.sh
   ```

   > _Recommended_: Ubuntu server with 128 GB RAM CPU instance.

3. **Download Models & Data**

   - Models & datasets: https://huggingface.co/Vorakorn1001/FinRL-DeepSeek-Models/tree/main

---

## 📂 Data & Agents

### 1. FNSPID Dataset

- **Source**: `Stock_news/nasdaq_exteral_data.csv` from [FNSPID on Hugging Face](https://huggingface.co/datasets/Zihan1004/FNSPID)
- **Reference**:
  - GitHub: https://github.com/Zdong104/FNSPID_Financial_News_Dataset
  - Paper: https://arxiv.org/abs/2402.06698

### 2. FinRL-DeepSeek Dataset

- **Source**: https://huggingface.co/datasets/benstaf/nasdaq_2013_2023/tree/main

### 3. FinRL-DeepSeek Trading Agents

- **Pretrained Agents**: https://huggingface.co/benstaf/Trading_agents/tree/main

---

## 🏋️ Training

### Mini-Model Training

- **PPO (mini)**

  ```bash
  nohup ./train_ppo_mini.sh > logs/train_ppo_mini.log 2>&1 &
  ```

- **CPPO-DeepSeek (mini)**

  ```bash
  nohup ./train_cppo_llm_risk_mini.sh > logs/train_cppo_llm_risk_mini.log 2>&1 &
  ```

Monitor training progress in log files for metrics such as `AverageEpRet`, `KL`, and `ClipFrac`.

### Environments

- **env_stocktrading.py**: PPO & CPPO baseline
- **env_stocktrading_llm.py / env_stocktrading_llm_01.py**: PPO-DeepSeek
- **env_stocktrading_llm_risk.py / env_stocktrading_llm_risk_01.py**: CPPO-DeepSeek

---

## 📊 Evaluation

Evaluate all agents with:

```bash
python evaluate.py
```

The evaluation script reports:

- **Information Ratio**
- **Conditional Value at Risk (CVaR)**
- **Rachev Ratio**
- **Daily & Weekly Outperformance Frequency**

Compare metrics across models to assess adaptive selection performance.

---

## 📚 References

- **FinRL-DeepSeek**: https://github.com/benstaf/FinRL_DeepSeek?tab=readme-ov-file
- **FinRL Contest 2025**: https://open-finance-lab.github.io/FinRL_Contest_2025/
- **FinRL-DeepSeek Paper**: https://arxiv.org/abs/2502.07393

