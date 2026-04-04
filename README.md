# Reinforcement Learning for Data Scientists
## From Intuition to LLMs — Companion Code Repository

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/rl-for-data-scientists/blob/main/notebooks/chapter_01_what_is_rl/01_rl_intuition.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

This repository contains all companion notebooks, exercises, and scripts for the book **"Reinforcement Learning for Data Scientists: From Intuition to LLMs"**.

Every chapter has one or more Jupyter notebooks with:
- **Concept notebooks**: annotated implementations of the chapter's key algorithms
- **Exercise notebooks**: hands-on problems with solutions in a companion file
- **Colab links**: one-click execution with no local setup required

---

## 🗂️ Repository Structure

```
rl-for-data-scientists/
├── notebooks/
│   ├── chapter_01_what_is_rl/          Chapter 1: What Is RL?
│   ├── chapter_02_prerequisites/        Chapter 2: Prerequisites
│   ├── chapter_03_mdps/                Chapter 3: Markov Chains & MDPs
│   ├── chapter_04_policies_values/      Chapter 4: Policies, Values & Rewards
│   ├── chapter_05_q_learning/          Chapter 5: Model-Free RL & Q-Learning
│   ├── chapter_06_policy_gradients_ppo/ Chapter 6: Policy Gradients & PPO
│   ├── chapter_07_recommender/         Chapter 7: RL for Recommendation
│   ├── chapter_08_hpo/                 Chapter 8: RL for HPO & AutoML
│   ├── chapter_09_bandits_offline/     Chapter 9: Bandits & Offline RL
│   ├── chapter_10_llm_agents/          Chapter 10: LLM Agents
│   ├── chapter_11_agent_training/      Chapter 11: Training Agents with RL
│   ├── chapter_12_when_to_use_rl/      Chapter 12: When to Use RL
│   ├── chapter_13_llm_pipeline/        Chapter 13: How LLMs Are Trained
│   ├── chapter_14_reward_model/        Chapter 14: Reward Modeling
│   ├── chapter_15_rlhf_trl/            Chapter 15: RLHF with TRL
│   ├── chapter_16_dpo/                 Chapter 16: DPO
│   ├── chapter_17_grpo/                Chapter 17: GRPO & DeepSeek
│   └── chapter_18_sql_project/         Chapter 18: End-to-End SQL Project
├── utils/                              Shared utility functions
├── data/                               Sample datasets
├── scripts/
│   ├── environment_setup.sh            One-command local setup
│   └── colab_setup.py                  Colab-compatible setup
└── .github/workflows/                  CI: notebook smoke tests
```

---

## 🚀 Quick Start

### Option 1: Google Colab (No Setup Required)

Click any **"Open in Colab"** badge in the notebook table below. Everything runs in the cloud — free T4 GPU available.

### Option 2: Local Setup

```bash
git clone https://github.com/yourusername/rl-for-data-scientists.git
cd rl-for-data-scientists
bash scripts/environment_setup.sh
jupyter lab
```

### Option 3: Conda Environment

```bash
conda create -n rl-book python=3.10
conda activate rl-book
pip install -r requirements.txt
jupyter lab
```

---

## 📚 Notebooks by Chapter

| Chapter | Topic | Notebooks | Colab | GPU? |
|---------|-------|-----------|-------|------|
| Ch 1 | What Is RL? | `01_rl_intuition.ipynb`, `01_exercises.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) | ❌ CPU |
| Ch 2 | Prerequisites | `02_probability_review.ipynb`, `02_exercises.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) | ❌ CPU |
| Ch 3 | MDPs | `03_markov_chains.ipynb`, `03_mdp_gridworld.ipynb`, `03_exercises.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) | ❌ CPU |
| Ch 4 | Policies & Values | `04_bellman_equations.ipynb`, `04_policy_iteration.ipynb`, `04_exercises.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) | ❌ CPU |
| Ch 5 | Q-Learning & DQN | `05_q_learning_scratch.ipynb`, `05_dqn_cartpole.ipynb`, `05_exercises.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) | ⚡ Optional |
| Ch 6 | Policy Gradients & PPO | `06_reinforce.ipynb`, `06_ppo_scratch.ipynb`, `06_dpo_scratch.ipynb`, `06_exercises.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) | ⚡ Optional |
| Ch 7 | RL for Recommenders | `07_dqn_recommender.ipynb`, `07_exercises.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) | ⚡ Optional |
| Ch 8 | RL for HPO | `08_reinforce_xgboost_hpo.ipynb`, `08_exercises.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) | ❌ CPU |
| Ch 9 | Bandits & Offline RL | `09_mab_thompson_ucb.ipynb`, `09_contextual_bandit_linucb.ipynb`, `09_exercises.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) | ❌ CPU |
| Ch 10 | LLM Agents | `10_react_agent.ipynb`, `10_agent_loop_mdp.ipynb`, `10_exercises.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) | ❌ CPU |
| Ch 11 | Agent RL Training | `11_gae_implementation.ipynb`, `11_code_agent_rlvr.ipynb`, `11_exercises.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) | 🔥 T4/A100 |
| Ch 12 | When to Use RL | `12_decision_framework.ipynb`, `12_benchmarking_agents.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) | ❌ CPU |
| Ch 13 | LLM Training Pipeline | `13_pretraining_demo.ipynb`, `13_sft_demo.ipynb`, `13_token_as_action.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) | ⚡ Optional |
| Ch 14 | Reward Modeling | `14_preference_dataset.ipynb`, `14_reward_model_training.ipynb`, `14_reward_hacking_demo.ipynb`, `14_exercises.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) | ⚡ T4 |
| Ch 15 | RLHF with TRL | `15_ppo_rlhf_trl.ipynb`, `15_failure_modes.ipynb`, `15_exercises.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) | 🔥 T4/A100 |
| Ch 16 | DPO | `16_dpo_from_scratch.ipynb`, `16_dpo_trl.ipynb`, `16_dpo_vs_ppo.ipynb`, `16_exercises.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) | 🔥 T4/A100 |
| Ch 17 | GRPO & DeepSeek | `17_grpo_from_scratch.ipynb`, `17_rlvr_math.ipynb`, `17_exercises.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) | 🔥 A100 |
| Ch 18 | SQL End-to-End Project | `18_01_sft_sql.ipynb`, `18_02_reward_model_sql.ipynb`, `18_03_ppo_sql.ipynb`, `18_04_evaluation.ipynb`, `18_05_dpo_comparison.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) | 🔥 A100 |

**GPU Legend:** ❌ CPU only · ⚡ GPU optional (faster) · 🔥 GPU strongly recommended

---

## 🛠️ Hardware Guide

| Chapters | Minimum Hardware | Recommended | Colab Tier |
|----------|-----------------|-------------|------------|
| 1–9 | CPU | CPU | Free |
| 10–12 | CPU | CPU | Free |
| 13–14 | CPU or T4 | T4 GPU | Free |
| 15–16 | T4 GPU | A100 GPU | Pro |
| 17 | A100 GPU | A100 GPU | Pro+ |
| 18 (Full) | T4 GPU | A100 GPU | Pro |

---

## 📦 Requirements

```
torch>=2.0.0
transformers>=4.40.0
trl>=0.8.0
peft>=0.10.0
datasets>=2.18.0
accelerate>=0.28.0
bitsandbytes>=0.43.0
gymnasium>=0.29.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
sqlparse>=0.4.4
sqlglot>=23.0.0
sympy>=1.12
scikit-learn>=1.3.0
```

Full pinned requirements: `requirements.txt` | `requirements-dev.txt`

---

## 🤝 Contributing

Found a bug? Have an improvement? Want to add your domain-specific adaptation?

1. Fork the repository
2. Create a feature branch: `git checkout -b fix/chapter-05-typo`
3. Make changes and test: `pytest tests/`
4. Submit a pull request

See `CONTRIBUTING.md` for full guidelines.

---

## 📄 License

MIT License — see `LICENSE` for details. The book text is © the author. The code in this repository is freely usable and modifiable.

---

## 📬 Contact

- **Issues**: Use GitHub Issues for bug reports and questions
- **Discussions**: Use GitHub Discussions for broader questions about the material
