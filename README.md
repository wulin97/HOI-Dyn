# HOI-Dyn

**NeurIPS 2025**  
**HOI-Dyn: Learning Interaction Dynamics for Human-Object Motion Diffusion**

## Abstract

Generating realistic 3D human-object interactions (HOIs) remains a challenging task due to the difficulty of modeling detailed interaction dynamics. Existing methods treat human and object motions independently, resulting in physically implausible and causally inconsistent behaviors. In this work, we present **HOI-Dyn**, a novel framework that formulates HOI generation as a **driver-responder system**, where human actions drive object responses. At the core of our method is a lightweight transformer-based interaction dynamics model that explicitly predicts how objects should react to human motion.

To further enforce consistency, we introduce a **residual-based dynamics loss** that mitigates the impact of dynamics prediction errors and prevents misleading optimization signals. The dynamics model is used only during training, preserving inference efficiency.

Through extensive qualitative and quantitative experiments, we demonstrate that our approach not only enhances the quality of HOI generation but also establishes a feasible metric for evaluating the quality of generated interactions.

> This code is based on [CHOIS (ECCV 2024)](https://github.com/lijiaman/chois_release). We sincerely thank the authors for releasing their code.

---

## Overview

This repository provides the interaction dynamics component of **HOI-Dyn**.

The code is designed as a lightweight module built on top of the original CHOIS codebase. After setting up CHOIS, users only need to place this folder into the main CHOIS directory to train and evaluate the HOI-Dyn dynamics model.

The released code supports:

1. Training the interaction dynamics model.
2. Testing the trained dynamics model.
3. Running autoregressive dynamics evaluation.
4. Using a trained dynamics model to supervise an HOI generation model during training.

The dynamics model predicts object responses conditioned on human motion, object geometry, and contact information. It can also be used as a training-time regularizer for diffusion-based HOI generation by computing a dynamics-consistency loss on the predicted clean HOI sample \(x_0\).

The dynamics model is only used during training and does not introduce extra inference cost for the final HOI generator.

---

## Code Structure

The folder contains:

```text
HOI-Dyn/
├── cfg/
│   └── test.yaml
├── core/
│   ├── loss.py
│   ├── model.py
│   ├── tools.py
│   ├── trainer.py
│   └── utils.py
└── README.md
```

Main files:

```text
core/model.py      # Dynamics model and dynamics-loss API for generated HOI
core/loss.py       # Dynamics loss
core/trainer.py    # Training / validation / testing logic
core/tools.py      # Data and geometry utilities
core/utils.py      # Utility functions
cfg/test.yaml      # Example config
```

The function for applying the trained dynamics model to generated HOI sequences is implemented in:

```text
core/model.py
```

Specifically:

```python
compute_dynamics_loss_for_generated_hoi(...)
```

---

## Installation

Please first install and prepare the original CHOIS codebase:

```text
https://github.com/lijiaman/chois_release
```

After CHOIS is correctly installed, place this HOI-Dyn folder under the main CHOIS directory.

For example:

```text
chois_release/
├── manip/
├── data/
├── checkpoints/
├── HOI-Dyn/
│   ├── cfg/
│   ├── core/
│   └── README.md
└── ...
```

This code relies on the data format, dependencies, and preprocessing pipeline of CHOIS.

---

## Training the Dynamics Model

Edit the config file:

```text
HOI-Dyn/cfg/test.yaml
```

Then run the following command from the CHOIS main directory:

```bash
python HOI-Dyn/core/trainer.py --cfg HOI-Dyn/cfg/test.yaml
```

The checkpoints will be saved to:

```text
./runs/<exp_name>/best.pth
./runs/<exp_name>/current.pth
```

---

## Testing the Dynamics Model

To test a trained dynamics model, set the config mode to:

```yaml
mode: test
ckpt: ./runs/hoidyn_dynamics/best.pth
```

Then run:

```bash
python HOI-Dyn/core/trainer.py --cfg HOI-Dyn/cfg/test.yaml
```

The script reports the dynamics prediction loss, including the world-space object point-cloud error.

---

## Autoregressive Evaluation

The dynamics model can also be evaluated autoregressively to inspect error accumulation over time.

Set:

```yaml
mode: test_ar
ckpt: ./runs/hoidyn_dynamics/best.pth
```

Then run:

```bash
python HOI-Dyn/core/trainer.py --cfg HOI-Dyn/cfg/test.yaml
```

This mode rolls out object motion step by step using previous predictions as input history.

---

## Using the Dynamics Model for HOI Generator Training

A trained dynamics model can be used to guide the training of an HOI generation model.

In HOI-Dyn, the diffusion model first predicts the clean HOI representation \(x_0\). Then the frozen dynamics model computes a dynamics-consistency loss on this predicted \(x_0\).

The core API is implemented in:

```text
core/model.py
```

Function:

```python
compute_dynamics_loss_for_generated_hoi(...)
```

The dynamics model should be frozen when used in this way. The function internally calls:

```python
self.eval()
```

to disable dropout or batch-normalization behavior, while still allowing gradients to flow back to the generated HOI input.

---


---

## Important Notes

- This repository is not a standalone replacement for CHOIS.
- Please first install and prepare CHOIS.
- Place this folder under the CHOIS main directory.
- The dynamics model uses the CHOIS-style processed HOI data.
- The trained dynamics model is used during HOI-generator training only.
- The final HOI generator does not require the dynamics model during inference.
- The dynamics loss should be applied after the diffusion model predicts \(x_0\), rather than on noisy diffusion states.

---

## Acknowledgement

This code is based on [CHOIS](https://github.com/lijiaman/chois_release). We thank the CHOIS authors for their excellent work and for making their implementation publicly available.

---

## Citation

If you find this code useful, please consider citing:

```bibtex
@article{wu2026hoi,
  title={Hoi-dyn: Learning interaction dynamics for human-object motion diffusion},
  author={Wu, Lin and Chen, Zhixiang and Lan, Jianglin},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={90795--90825},
  year={2026}
}
```

Please also cite CHOIS if you use the original CHOIS codebase or data-processing pipeline.