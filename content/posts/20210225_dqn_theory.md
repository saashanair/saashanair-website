---
title: "Deep Q Networks (DQN): Theory"
date: 2021-02-25
tags: "Reinforcement Learning"
math: true
---

**TL;DR:** DQN is an off-policy, value-based, model-free RL algorithm, that learns to act in discrete action spaces.

---

This is the first post in a four-part series on DQN.

- Part 1: The components of the algorithm
- {{< backlink "20210305_dqn_algo_to_code" "Part 2">}}: Translating algorithm to code
- {{< backlink "20210321_dqn_and_its_hyperparams" "Part 3">}}: Effects of the various hyperparameters
- {{< backlink "20210411_double_dqn" "Part 4">}}: Combating overestimation with Double DQN

---

{{< toc >}}

---

Hey there dear reader, hope you have been having a good week. 🤗  I apologise for the delay in publishing this week's post. 

DQN is quite an important algorithm in Deep RL. It lays the foundation for the field, with the principles introduced in the paper being used even today. Its success in using Deep Neural Networks to perform well across a range of environments causes it to be often dubbed as the "ImageNet of Deep RL".

DQN was first introduced in 2013, but the Nature-variant published in 2015 is the one we discuss in this post. It introduced two key components:

1. Replay Memory
2. Target Network

## Replay Memory

This component is often referred to by a variety of names such as 'Replay Buffer', 'Experience Replay' and the likes. The concept behind this is extremely simple and [has been around since the early 90s](http://isl.anthropomatik.kit.edu/pdf/Lin1993.pdf). "So what exactly is this component?"

Have you had one of those conversations with someone, where long after the conversation is over, you replay it over and over again in your mind, thinking if you could redo it, how you would respond differently this time. Replay Buffer can be thought of as the digital equivalent of this for a DQN agent (and also {{< backlink "20210201_ddpg_theory" "DDPG">}} and {{< backlink "20210112_td3_theory" "TD3">}}). The idea is that it helps reinforce what was right and what went wrong in the exchange, enabling you and the DQN agent to be better prepared should a similar situation arise in the future. "How does this help the DQN agent?"

Replay Buffer makes DQN [sample efficient](https://ai.stackexchange.com/questions/5246/what-is-sample-efficiency-and-how-can-importance-sampling-be-used-to-achieve-it). Instead of throwing away experiences after the agent has learnt from them once, the experiences are re-used multiple times. This can help reduce the number of interactions required by the agent, and can be especially useful in scenarios where such interactions might be rare (e.g., spotting a sink hole while driving) or costly (e.g., being involved in a car crash leading to property damage or even death). This can also help combat [catastrophic forgetting](https://en.wikipedia.org/wiki/Catastrophic_interference) as the agent re-learns from past experiences.

Additionally, if we use data from only a single trajectory to learn, all the data points would be correlated and local to that part of the function. However, optimisation algorithms assume the data to be [independent and identically distributed (i.i.d)](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables). Thus, using a Replay Buffer can help break correlations and force the data to be more representative of the distribution (by sampling across time from different trajectories and different locations of the same trajectory).

"So how is this concept implemented?" The off-policy nature of DQN lends itself well to the use of a Replay Buffer. The idea is that each interaction that the agent has with the environment is stored in the buffer in the form of a 5-tuple of <state, action, next_state, reward, done>. At each time step then, a mini-batch of experiences are sampled uniformly at random for computing the gradient updates to allow the neural network to learn.

The Replay Buffer is a finite-sized array, in which past experiences are over-written with newer ones after a certain amount of time. We do not need to store the entire history of experiences across the agent's lifetime. If the buffer is too large, it would "over-replay" early experiences that had poor performance. On the other hand, if the memory is too small, the agent would over-index for recent experiences.

## Target Network

The strategy for learning in DQN (and {{< backlink "20210201_ddpg_theory" "DDPG">}} and {{< backlink "20210112_td3_theory" "TD3">}}) is to transform the problem into that of supervised learning. "But we do not have any labels", you might wonder. Yes, you are right. But this is where the recursive nature of the Bellman equation for Q-values helps us. As per the Bellman equation,

$Q(s, a) = r + \gamma \cdot max_{a\'} Q(s\', a\')$

Thus, the idea is to use the right side of the equation above as our 'true label' and the left side of the equation as the 'predicted label'. Similar to any supervised learning approach then, the aim is to bring 'predicted label' as close as possible to 'true label'. However, this naive implementation poses a problem. Since the 'true label' (hereafter referred to as the TD-Target) and the 'predicted label' are estimated from the same network, as the network improves the TD-Target also shifts. The effect of this is similar to a cat crazily chasing after a laser pointer.

<img src="https://steamuserimages-a.akamaihd.net/ugc/718665440849974354/DD00A39B23D1E5B7A21568F88AE0992E45D976DB/?imw=5000&imh=5000&ima=fit&impolicy=Letterbox&imcolor=%23000000&letterbox=false" class="large" alt="">
<em><a href="https://steamcommunity.com/sharedfiles/filedetails/?id=433115079">Source</a></em>


To combat this, the authors introduce Target Networks (yes the very same that we have used in the series on {{< backlink "20210201_ddpg_theory" "DDPG">}} and {{< backlink "20210112_td3_theory" "TD3">}} previously 😉). The idea is to have two copies of the Q-network, one is kept current by updating it at every time step, while the other is kept frozen for a set number of steps. This ensures that the target is not continuously moving. After every 'n' gradient update steps, the target network is updated by copying the parameters of the current network into the target. This modification was found to help improve the stability during training by reducing oscillations and divergence of the policy.

## The schematics

So the next obvious question is, "How does the agent learn?" DQN uses only two neural nets, the Q-network and its target.

<img src="/images/posts/20210225_dqn_theory/schematics.png" class="large" alt="">
<em>Fig 1: DQN uses a total of two networks: a Q-network and its target</em>


Q-value, mathematically denoted as $Q(s, a)$ is a measure of the "quality" or "goodness" of a particular state-action pair. More concretely, it represents the expected return of performing action $a$ in state $s$ and then following the policy $\pi$. Computing the Q-value requires both state and action information (as we had noticed in {{< backlink "20210201_ddpg_theory" "DDPG">}} and {{< backlink "20210112_td3_theory" "TD3">}}). However, DQN only takes the state vector as input. The output layer of the network then has the same shape as the action space of the environment, thus outputting the Q-values of each of the actions (this setup is the reason why DQN cannot be applied to a continuous action space). The agent then acts by picking the action with the highest Q-value.

<img src="/images/posts/20210225_dqn_theory/q-network.png" class="large" alt="">
<em>Fig 2: Architecture of the Q-network</em>


As for the loss computation, mean-squared-error is applied to the TD-Target and the 'predicted y' on a mini-batch of past experiences sampled from the Replay Buffer. Each of the experiences in the batch contains information about <state, action, next state, reward, done>. The 'state' and the 'action' from the sampled batch of experiences are used to compute the 'Predicted Q' using the Q-network. The 'next state' and the 'reward' from the sampled batch are involved in computing the 'Target Q' via the Target Network.

<img src="/images/posts/20210225_dqn_theory/loss.png" class="large" alt="">
<em>Fig 3: Block diagram depicting how the loss is computed in DQN</em>


## Exploration in DQN

To encourage the agent to explore the environment, the off-policy nature (i.e., behavioural policy is different from the policy being learnt) of the algorithm comes in handy.

The authors of the paper use a ***Annealing Epsilon-Greedy policy*** during training. This works by starting with a maximum epsilon value of 1.0 and over time gradually decreasing the value to 0.01. To select the action performed at each time step, you start by generating a random number, comparing it against the current epsilon value, if the random number is smaller, then perform an exploratory (i.e. random) action, else perform an exploitative action. This essentially means that at the start when epsilon is at 1.0, the agent performs purely exploratory actions, but as the epsilon value declines, so does the amount of exploration performed by the agent.

## The DQN Algorithm: Putting the pieces together

<img src="/images/posts/20210225_dqn_theory/algo.png" class="large" alt="">
<em>Fig 4: The DQN Algorithm</em>


## So what's next?

Alright, so that is it for today. Hope to see you again next week with the second part of this series. In the meantime, please feel free to reach out via email at [saasha.allthingsai@gmail.com](mailto:saasha.allthingsai@gmail.com) or via [Twitter](https://twitter.com/saasha_nair).

See you soon! 👩‍💻

---

## Further reading

1. [Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf), Volodymyr Mnih et. al. -- the 2015 Nature-variant of DQN
2. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf), Volodymyr Mnih et. al. -- the 2013 vanilla-DQN (uses only Replay Memory and no Target Network)
3. [Let's build a DQN: Basics](https://tomroth.com.au/dqn-basics/), Tom Roth -- a three-part series about how DQN works
4. [Reinforcement Learning for Robots Using Neural Networks](http://isl.anthropomatik.kit.edu/pdf/Lin1993.pdf), Long-Ji Lin -- Section 3.5 of the Thesis explains Experience Replay and its benefits
5. [RL Series #2: Learning with Deep Q Networks (DQN)](https://saashanair.medium.com/rl-series-2-dqn-e739eb3ab1d1) -- my previous post on DQN on Medium