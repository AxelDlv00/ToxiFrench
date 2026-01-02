## Basic Fine-tuning

Following [1] and [2]: 

Let $\mathcal D=\{(x_i, y_i)\}_{i=1}^N$ be a dataset of input-output pairs, where $x_i$ is the input sequence (the prompt) and $y_i$ is the corresponding target sequence (the answer). 

SFT minimizes the sentence-level cross-entropy loss:

$$\mathcal L_{SFT}(\theta) = \mathbb{E}_{(x,y) \sim \mathcal D}[-\log \pi_\theta(y|x)]$$

### Token-Level Decomposition

In causal language modeling, the probability $\pi_\theta(y|x)$ is decomposed into the product of conditional probabilities for each token in the response. 

Using the chain rule, we have:

$$ \pi_\theta(y|x) = \prod_{t=1}^{|y|} \pi_\theta(y_t | y_{<t}, x) $$

Therefore:

$$ \log \pi_\theta(y|x) = \sum_{t=1}^{|y|} \log \pi_\theta(y_t | y_{<t}, x) $$


During training, the loss is typically masked for the prompt $x$ and calculated only on target tokens $y_t$:

$$\mathcal L_{SFT}(\theta) = \mathbb{E}_{(x,y) \sim \mathcal D} \left[ - \sum_{t=1}^{|y|} \log \pi_\theta(y_t | y_{<t}, x) \right]\approx \frac{1}{|\mathcal D|} \sum_{(x,y)\in \mathcal D} \left[ - \sum_{t=1}^{|y|} \log \pi_\theta(y_t | y_{<t}, x) \right]$$

## Implementing DFT [2]

The loss used in DFT is defined as:

$$\mathcal L_{DFT}(\theta) = \mathbb{E}_{(x,y) \sim \mathcal D} \left[ - \sum_{t=1}^{|y|} sg(\pi_\theta(y_t| y_{<t}, x)) \log \pi_\theta(y_t | y_{<t}, x) \right]$$

> Note : The TRL library provides an implementation of DFT loss (https://huggingface.co/docs/trl/sft_trainer)

## Dynamic Weighted Loss Implementation

Assume we have $\mathcal C$ classes in the output that we want to assign different weights $\alpha_c$ to. For stability we impose that $\sum_{c=1}^{\mathcal C} \alpha_c = 1$.

During the training, there are $E$ epochs, each epoch consisting of $S$ steps (batches). Therefore, the training is decomposed into ordered steps $\tau_{e,s}$ where $e \in \{1, \ldots, E\}$ and $s \in \{1, \ldots, S\}$ which we will denote by $\tau$ for simplicity.

These weights can be dynamically adjusted during training, therefore $\alpha_c = \alpha_c(\tau)$.

The dynamic weighted loss can be expressed as:
$$\mathcal L_{DWL}(\theta, \tau) = \mathbb{E}_{(x,y) \sim \mathcal D} \left[ - \sum_{t=1}^{|y|} \left( \sum_{c=1}^{\mathcal C} \alpha_c(\tau) \cdot \mathbb{1}_{y_t \in c} \right) \log \pi_\theta(y_t | y_{<t}, x) \right]$$

This can be written as : 

$$ \mathcal L_{DWL}(\theta, \tau) = \sum_{c\in \mathcal C} \alpha_c(\tau) \cdot \mathcal L_c(\theta) $$

where
$$ \mathcal L_c(\theta) = \mathbb{E}_{(x,y) \sim \mathcal D} \left[ - \sum_{t=1}^{|y|} \mathbb{1}_{y_t \in c} \log \pi_\theta(y_t | y_{<t}, x) \right] $$

---

### Sources
* [1] Radford, A. (2018). *Improving language understanding by generative pre-training*. [cite: 1]
* [2] Wu, Y., Zhou, Y., et al. (2025). *On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification*.