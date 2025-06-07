---
title: "Deep Deterministic Policy Gradient (DDPG): Theory"
date: 2021-02-01
tags: "Reinforcement Learning"
math: true
---

**TL; DR:** Deep Deterministic Policy Gradient, or DDPG in short, is an actor-critic based off-policy reinforcement learning algorithm. It combines the concepts of Deep Q Networks (DQN) and Deterministic Policy Gradient (DPG) to learn a deterministic policy in an environment with a continuous action space.

---

NOTE1: This is the first post in a two-part series on DDPG.

- Part 1: What are the components of the algorithm and how do they work?
- {{< backlink "20210215_ddpg_algo_to_code" "Part 2">}}: How to translate the algorithm to code?

---

{{< toc >}}

---

The first image invoked on hearing the words 'Reinforcement Learning' is often the DQN algorithm. Despite its success in the Atari environments, DQN only works within discrete action space. Most real-world applications, such as robotics, autonomous driving and so on, however, require the actions to have continuous-values. Thus, Deep Deterministic Policy Gradient (DDPG) attempts to fill this gap by introducing a few "tweaks" that allow DQN to be extended to continuous action space.

## The schematics

First things first, how many neural networks do we require? DDPG is based on the off-policy deterministic actor-critic setting (read [this paper on Deterministic Policy Gradient Algorithms](https://twitter.com/saasha_nair) for more details). Thus, we require at least two networks, one for the actor and another for the critic.

Additionally, to ensure stability during training, DDPG borrows the concept of target networks from [DQN](https://medium.com/@saasha/rl-series-2-dqn-e739eb3ab1d1). Thus, we need a total of 4 networks to run this algorithm; an actor $\mu$ (also represented as $\pi$ sometimes, as in the [TD3 paper](https://arxiv.org/pdf/1802.09477.pdf)), a critic $Q$ and their respective targets, $\mu\'$ and $Q\'$.

<img src="/images/posts/20210201_ddpg_theory/total-networks.png" class="large" alt="">
<em>Fig 1: DDPG uses a total of 4 networks - one actor, one critic and their corresponding targets</em>

Being off-policy, DDPG also benefits from the use of a 'Replay Buffer', similar to DQN. The replay buffer is a finite-sized cache that stores past experiences from which batches are sampled uniformly at random for training.

## The DDPG Critic

As the name suggests, the task of the critic is to "critique" the actor's beliefs by determining 'how good is the suggested action given the current state'. In other words, the critic is tasked with computing the Q-value $Q(s,a)$ of the given state-action pair.

"Wait! So if we are in any case computing the Q-value, why not directly use DQN then?", you might wonder. Let's for a minute think back to how the Q-value was computed in [DQN](https://medium.com/@saasha/rl-series-2-dqn-e739eb3ab1d1). In the DQN setting, the state is given as input to the network, while at the other end we generate 'n' Q-values, one for each of the possible discrete actions. To select the action to be performed, we pick the action with the highest Q-value. Computing Q-values for all possible actions makes sense when we have a small number of discrete actions to pick from. But in situations where we need to train an RL agent to control a robotic arm or drive an autonomous vehicle, it isn't sufficient to just say 'turn left', we need to specify 'by how much' we need it to turn. We, thus, need a continuous action space to allow for such fine-grained control.

"So, how does DDPG compute the Q-value then?", you may ask. Since DDPG delegates the responsibility of predicting the action and determining the "goodness" of that action to two different networks, the critic must only compute the Q-value based on the action selected by the other network. Thus, the input to the critic, here, is the state and the predicted action, while the output layer of the network has a single neuron that produces the Q-value of the given state-action pair.

<img src="/images/posts/20210201_ddpg_theory/dqn-vs-ddpg.png" class="large" alt="">
<em>Fig 2: Calculation of Q-values in DQN vs DDPG Critic</em>

The next natural question is, "How does the critic network learn?" The Mean-Squared Bellman Error is used, similar to DQN. Thus, the loss is computed as the Mean-Squared Error between the TD-Target and the Q-value estimate of the current state and corresponding action. The TD-Target $y$ utilises the target networks. The next state $s\'$ and the associated action predicted by the target actor $\mu\'$ are provided as input to the target critic $Q\'$. This can be formulated as:

$y = r + \gamma \cdot Q\'(s\', \mu\'(s\'))$

<img src="/images/posts/20210201_ddpg_theory/critic-loss.png" class="large" alt="">
<em>Fig 3: Calculation of TD-Target that is used in the Critic Loss</em>


## The DDPG Actor

Being based on [DPG](http://proceedings.mlr.press/v32/silver14.pdf?CFID=6293331&CFTOKEN=eaaee2b6cc8c9889-7610350E-DCAB-7633-E69F572DC210F301), the DDPG agent learns a deterministic policy. This means that the actor-network learns to map a given state to a specific action, instead of a probability distribution on the actions, as is done in algorithms that learn a stochastic policy. Thus, the actor-network takes as input the state vector and outputs the action to be performed, with the size of the layer dependent on the size of the action space.

The actor-network improves based on the "critique" of the critic network. Thus, based on the Deterministic Policy Gradient Theorem (which we will get into at a later date), the actor-network learns by performing a gradient ascent (thereby requiring the negative sign indicated below for implementation with autodiff), with respect to the policy, along the direction indicated by the critic. This can be formulated as:

$L_{\text{actor}} = - Q(s, \mu(s))$

<img src="/images/posts/20210201_ddpg_theory/actor-loss.png" class="large" alt="">
<em>Fig 4: Calculation of the Actor Loss</em>

## Updating the DDPG Targets

Target networks were introduced in DDPG to deal with the instability caused due to the use of the Bellman Error. Since the critic updates in DDPG depend on the Bellman Error, target networks become necessary. However, the two algorithms differ in the way that the targets are updated.

DQN works by applying hard updates, i.e., periodically the weights of the main network are copied into the target network directly. In contrast, DDPG performs 'soft updates' at each training step. As indicated below, Polyak averaging is used for this:

$\theta\' \leftarrow \tau \cdot \theta + (1 - \tau) \cdot \theta\'$

$\phi\' \leftarrow \tau \cdot \phi + (1 - \tau) \cdot \phi\'$

Here, $\theta$ and $\theta\'$ represent the parameters of the critic and the target critic networks respectively, while $\phi$ and $\phi\'$ represent the parameters of the actor and its target. The hyperparameter $\tau \in [0, 1]$ is set to an extremely small value to ensure that the targets update very slowly.

## Exploration in DDPG

The deterministic nature of the learned policy requires DDPG to implement a strategy for ensuring that the network does not get stuck in a local minimum during training. Since DDPG learns off-policy, we can encourage exploration by using a stochastic behaviour policy. This is implemented by adding a small amount of noise to the actions predicted by the actor-network during training, and can be formulated as:

$a = \mu(s) + \epsilon$

<img src="/images/posts/20210201_ddpg_theory/exploration-noise.png" class="large" alt="">
<em>Fig 5: Action prediction during training vs testing</em>


As per [the paper](https://arxiv.org/pdf/1509.02971.pdf), the choice of noise $\epsilon$ depends on the environment. For the robotics environments used in the paper, the authors suggest using a time-correlated noise generated using the [Ornstein-Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process). However, the [paper on TD3](https://arxiv.org/pdf/1802.09477.pdf) notes that applying a small Gaussian noise is often sufficient.

## The DDPG algorithm: Putting all the pieces together

Now that we have inspected each of the individual components, it is time to look at the algorithm as a whole.

<img src="/images/posts/20210201_ddpg_theory/ddpg-algo.png" class="large" alt="">
<em>Fig 6: The DDPG Algorithm</em>

## Roundup

With the theory behind DDPG out of the way, I urge you, dear reader, to think about how you would implement this algorithm. Do note though, that there are a few nuances relating to the implementation of this algorithm that we haven't yet discussed. We would be looking at these in the next post in this series on DDPG.

As always, thank you for stopping by! Hope you found this post useful. I love hearing from you, so please do feel free to get in touch either via [email](mailto:saasha.allthingsai@gmail.com) or via [Twitter](https://twitter.com/saasha_nair).

See you soon! 🤓

---

## Further reading

1. [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf), Timothy Lillicrap et. al. -- the paper on DDPG
2. [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf), David Silver et. al. -- the paper on DPG, along with its [supplementary material](http://proceedings.mlr.press/v32/silver14-supp.pdf)
3. [Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html), OpenAI Spinning Up -- explains how DDPG works and represents the algorithm in an implementation-friendly manner
4. [Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html), by Lilian Weng -- a long post on policy gradient methods, with a section on DDPG and TD3
5. [RL Series #2: Learning with DQN](https://medium.com/@saasha/rl-series-2-dqn-e739eb3ab1d1) -- my Medium post on DQN (or the [updated version on this site](https://www.saashanair.com/dqn-theory/))