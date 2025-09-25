# EVALUESTEER
---

This repository contains the implementation framework, Code and Data for the paper "EVALUESTEER: Measuring Reward Model Steerability Towards Values and Preferences" under review at ICLR 2026. It provides a skeleton implementation with all the core logic for reproducibility purposes.

## Overview

The framework consists of two main components:

### 1. Synthetic Data Generation (`synthetic_data_generation/`)
- **Purpose**: Generates synthetic user data with diverse value profiles and style preferences
- **Key Files**:
  - `user_data_generation.py`: Main data generation script
  - `wvs_user_profile_generation/`: World Values Survey-based profile generation
  - Data outputs: User profiles, synthetic questions, and style variations

### 2. Reward Model Evaluation (`reward_model_evaluation/`)
- **Purpose**: Systematic evaluation framework for preference prediction models
- **Key Components**:
  - `data_management/data_manager_v3.py`: Data loading and management
  - `evaluation_engine/evaluation_engine_v3.py`: Core evaluation logic
  - `reward_model_evaluation_*.py`: Evaluation scripts for different model types
  - `analysis/`: Result analysis frameworks

## Data Dependencies

**Required**: Download the IndieValue dataset from https://github.com/liweijiang/indievalue into this repository root directory before running any experiments.

```bash
git clone https://github.com/liweijiang/indievalue.git
```

## Architecture

The framework supports evaluation across multiple dimensions:
- **User Profiles**: Systematic combinations of value and style preferences
- **Model Types**: vLLM, OpenAI API, and sequence classifier models
- **Evaluation Settings**: Simple prompting, contextual prompting, and chain-of-thought reasoning
- **Analysis**: Comprehensive result analysis

## Note

This is a skeleton repository containing the core logic and architecture for reproducibility. It is not directly executable without proper setup of dependencies and data preprocessing. The implementation serves as a reference for the methodology described in the ICLR submission.
