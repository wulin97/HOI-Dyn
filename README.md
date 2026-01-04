# HOI-Dyn
NeurIPS'2025 HOI-Dyn: Learning Interaction Dynamics for Human-Object Motion Diffusion

# Abstract
Generating realistic 3D human-object interactions (HOIs) remains a challenging task due to the difficulty of modeling detailed interaction dynamics. Existing methods treat human and object motions independently, resulting in physically implausible and causally inconsistent behaviors. In this work, we present HOI-Dyn, a novel framework that formulates HOI generation as a driver-responder system, where human actions drive object responses. At the core of our method is a lightweight transformer-based interaction dynamics model that explicitly predicts how objects should react to human motion. 
To further enforce consistency, we introduce a residual-based dynamics loss that mitigates the impact of dynamics prediction errors and prevents misleading optimization signals. The dynamics model is used only during training, preserving inference efficiency. 
Through extensive qualitative and quantitative experiments, we demonstrate that our approach not only enhances the quality of HOI generation but also establishes a feasible metric for evaluating the quality of generated interactions.


> This code is based on [CHOIS (ECCV 2024)](https://github.com/lijiaman/chois_release).