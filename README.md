# RL-SEG

Project developed during an internship at INESC-TEC in 2023/2024.

## Summary

**Theme:** Resource efficiency using reinforcement learning for autonomous driving

**Objective**: This project ultimately targets the segmentation component of autonomous driving, implementing a reinforcement learning approach to iteratively segment image areas according to their importance for the decision-making process. Several images are extracted from the [BDD100K](https://bair.berkeley.edu/blog/2018/05/30/bdd/) dataset, where experimentation is performed, and multiple approaches using the PPO algorithm are tested, mainly targeting different reward functions.

## Repository Structure

```text
├── Introduction        # Introductory Python notebooks to the PyTorch deep learning library
├── Project             # Main project directory, containing the RL agent source code (Agent), an output directory with some example outputs from the experimentation (Outputs), and the segmentation source code and sample outputs (Segmentation)
├── RL-PPO              # Implementation of the PPO algorithm
└── RL-Snake            # Implementation of Q-Learning
