---
title: "Double DQN: A variant of DQN"
date: 2021-04-11
tags: "Reinforcement Learning"
math: true
---

**TL;DR:** Double DQN was introduced as a technique to deal with issues of overestimation in DQN. The benefits, however, are not as clear as described in the [paper](https://arxiv.org/pdf/1509.06461.pdf). The performance of Double DQN, when compared to DQN, can vastly vary based on the environment and hyperparameters.

---

This is the fourth post in a four-part series on DQN.

- {{< backlink "20210225_dqn_theory" "Part 1">}}: Components of the algorithm
- {{< backlink "20210305_dqn_algo_to_code" "Part 2">}}: Translating algorithm to code
- {{< backlink "20210321_dqn_and_its_hyperparams" "Part 3">}}: Effects of the various hyperparameters
- Part 4: Combating overestimation with Double DQN

---

{{< toc >}}

---

DQN has been a seminal piece of work for the Deep RL community. However, it suffers from some shortcomings. Over the years, researchers have introduced improvements over the algorithm. In this post, let's discuss one such early fix introduced in 2015, [Double DQN.](https://arxiv.org/pdf/1509.06461.pdf)

## Why Double DQN?

Recall the update step in the DQN algorithm. The neural network that backs the agent is updated by applying the loss function to the predicted and the target Q. However, the RL setting does not have a "ground truth" per se. Thus, target Q is estimated using the Bellman equation as:

TD-Target, $y_j = r_j + \gamma \cdot \text{max}{a\'} \hat{Q}(s_{j+1}, a\'; \theta^{-})$

$\hat{Q}$ in the equation above is the target network's estimate of the Q-values of performing each of the possible actions in the next state, $s_{j+1}$. The TD-Target is then computed by picking the highest Q-value, which is equivalent to picking the best possible action in state $s_{j+1}$. Therein lies the problem!

It is essential to understand that $\hat{Q}$ is just an estimate, and so it is error-prone. This means some of the estimated values might be much higher than the ground truth and some much lower. Since the true Q-values are unknown, we cannot fix these errors. Applying a max operator to such erroneous estimates results in the errors being further propagated. Since this max operator picks the highest Q-value, the resulting Q-estimates are likely to be higher than the true Q-values. This phenomenon is, thus, known as ***'overestimation bias'***. If this bias is encoded in an agent, it learns sub-optimal policies, which lead to poor performance. Thus, the need to tackle this issue.

Of course, overestimation bias is not unique to DQN. In fact, the TD3 algorithm was introduced to combat overestimation in DDPG. If you would like to read more about it, check out this post that deals with an {{< backlink "20210112_td3_theory#what-is-overestimation-bias" "intuitive explanation of overestimation bias in the post on TD3">}}.

## Double DQN: The Idea

The [Double DQN paper](https://arxiv.org/pdf/1509.06461.pdf) suggests decomposing the max operation in the equation used to compute the TD-Target (as seen above). Thus, instead of picking $a'$ and computing $\hat{Q}$ with the same set of weights $\theta^{-}$, the two steps are now decoupled. Double DQN estimates TD-Target by first selecting the best action $a\'$ in the given next state $s_{j+1}$ using the policy network. The value $\hat{Q}$ of this target action $a\'$ in state $s_{j+1}$ is then evaluated using the target network. Thus, the equation for computing the TD-Target is updated from:

TD-Target, $y_j = r_j + \gamma \cdot \text{max}{a\'} \hat{Q}(s{j+1}, a\'; \theta^{-})$

to:

TD-Target, $y_j = r_j + \gamma \cdot \hat{Q}(s_{j+1}, \text{argmax}{a} Q(s{j+1}, a; \theta); \theta^{-})$

## Double DQN: Implementation

The question then is, "What does this update in the equation mean for the code?" Using our PyTorch implementation of DQN from {{< backlink "20210305_dqn_algo_to_code#dqnagent" "part 2 of this series">}}, we only need to modify one line in the "learn" function of the DQNAgent class.

```python
## the following code is placed within the DQNAgent class
def learn(self, batchsize):
        """
        Function to perform the updates on the neural network that runs the DQN algorithm.

        Parameters
        ---
        batchsize: int
            Number of experiences to be randomly sampled from the memory for the agent to learn from

        Returns
        ---
        none
        """

        # select n samples picked uniformly at random from the experience replay memory, such that n=batchsize
        if len(self.memory) < batchsize:
            return
        states, actions, next_states, rewards, dones = self.memory.sample(batchsize, self.device)
        actions = actions.view(-1, 1)
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)

        # get q values of the actions that were taken, i.e calculate qpred
        q_pred = self.policy_net(states).gather(1, actions)

        # q_target = self.target_net(next_states).max(dim=1).values ## ORIGINAL

        ## CHANGE
        # calculating TD-Target: evaluate the greedy policy according to the online (policy) network, then using the target network to estimate its value.
        # select the target action using the policy network
        target_action = torch.argmax(self.policy_net(next_states), dim=1).detach()
        # estimate the Q value of the target action using the target network
        q_target = self.target_net(next_states).gather(1, target_action.view(-1, 1))
				## END OF CHANGE

        q_target[dones] = 0.0 # setting Q(s\',a\') to 0 when the current state is a terminal state
        y_j = rewards + (self.discount * q_target)

        # calculate the loss as the mean-squared error of yj and qpred
        self.policy_net.optimizer.zero_grad()
        loss = F.mse_loss(y_j, q_pred).mean()
        loss.backward()
        self.policy_net.optimizer.step()
```

## DQN vs Double DQN: Results

Is Double DQN really worth all the fuss? Let's find out!

For a fair comparison, the Double DQN and DQN agents have been trained with the same hyperparameters and network structure.

<img src="/images/posts/20210411_double_dqn/test_reward.png" class="large" alt="">
<em>Fig 1: Test time performance of DQN (left) and Double DQN (right) agents trained for 2000 episodes with update frequencies set to 10, 100 and 1000.</em>


In the previous post, we observed the performance of the DQN agent in the LunarLander-v2 environment. The DQN agent performed best with update frequencies of 10 and 1000. However, an update frequency of 100 caused the DQN agent to exhibit noisy test time performance. Plotting similar test time performance with Double DQN, we notice a drop in performance. An update frequency of 10 or 1000 causes the Double DQN to exhibit noisy test time performance, with the mean score dropping below the desired threshold for the environment at multiple points. Interestingly, despite still being sub-optimal, the Double DQN agent (right) with an update frequency of 100 appears to perform better than the corresponding DQN agent (left).

The training curves of these agents could give us some insights into how Double DQN compares against DQN. So, let's inspect those.

But wait, before we jump into it, let's try to understand the significance of the two types of graphs we will encounter below. The two things we are interested in observing here are:

1. Does Double DQN result in lower target Q-values than DQN?
2. How does the training reward correspond to the Q-values?

In all our code on this blog, to date, we have always worked with episodes. But this post will benefit from looking at individual timesteps instead. Thus, all the agents discussed below are trained for 600,000 timesteps.

To answer the first question about the effect on target Q-values, we plot the mean Q-value vs step plot. This is obtained by tracking the mean target Q-value of the batch of 64 samples at each training step. The idea here is that the lower the target Q-value, the lower the overestimation bias encoded in the agent.

The question about the training reward is studied by plotting the cumulative reward at each timestep. Thus, we track the running total reward of the agent. At each training step, the reward received by the agent is added to this running total.

In the graphs below, the DQN and the Double DQN agents are depicted in light grey and light purple, respectively. The region of overlap between the two curves is visible in a darker shade of purple.

### DQN vs DDQN: Update frequency of 10

<img src="/images/posts/20210411_double_dqn/upd_10.png" class="large" alt="">
<em>Fig 2: Performance of the DQN and Double DQN agents with an update frequency of 10 plotted per training timestep for a total of 600,000 steps. Left: Graph showing the mean Q-value of the two agents at each training step over a batch of 64 samples. Lower mean Q-values are preferable. The region of overlap between the mean Q-values of the two agents is visible in a darker shade of purple. Right: Total running reward of the agent plotted at each training step. Higher rewards are preferable.</em>


As per the graph on the left, the Double DQN agent's mean Q-value overshoots that of the DQN agent considerably early. Though there appears to be a glimmer of hope around the 300,000-th step, it fades quite quickly. For most of the training period, the mean Q-value of the Double DQN agent is much higher than the DQN agent.

We have previously discussed that a lower mean Q-value is desirable. Studying the graph on the right, we notice that owing to the lower mean Q-value, the cumulative reward received by the DQN agent is much higher. Also, note the peak in the curve corresponding to the Double DQN agent. The cumulative reward of the Double DQN agent rises around the 300,000-th mark, which corresponds to the short interval where its mean Q-values were lower than that of DQN.

### DQN vs Double DQN: Update Frequency 1000

<img src="/images/posts/20210411_double_dqn/upd_1000.png" class="large" alt="">
<em>Fig 3: Performance of the DQN and Double DQN agents with an update frequency of 1000 plotted per training timestep for a total of 600,000 steps. Left: Graph showing the mean Q-value of the two agents at each training step over a batch of 64 samples. Lower mean Q-values are preferable. The region of overlap between the mean Q-values of the two agents is visible in a darker shade of purple. Right: Total running reward of the agent plotted at each training step. Higher rewards are preferable.</em>


With the update frequency set to 1000, the Double DQN agent appears to give the DQN agent a tough competition at the start. But then the performance of the Double DQN agent deteriorates quickly. After the first 70,000 odd steps, the Double DQN agent's mean Q-values are at least as high as that of the DQN agent, if not higher.

The cumulative reward graph (right) mimics this trend as well. The performance of the two agents seems quite competitive at the start. But then the Double DQN agent, though managing to keep up with the DQN agent (with the two curves running parallel), is constantly underperforming.

### DQN vs Double DQN: Update Frequency 100

<img src="/images/posts/20210411_double_dqn/upd_100.png" class="large" alt="">
<em>Fig 4: Performance of the DQN and Double DQN agents with an update frequency of 100 plotted per training timestep for a total of 600,000 steps. Left: Graph showing the mean Q-value of the two agents at each training step over a batch of 64 samples. Lower mean Q-values are preferable. The region of overlap between the mean Q-values of the two agents is visible in a darker shade of purple. Right: Total running reward of the agent plotted at each training step. Higher rewards are preferable.</em>


Finally, a curve that matches the promise of the Double DQN paper! There is a short interval around the 200,000-th mark where it seems like even this Double DQN agent might be going along the same path as the one above. But after that interval, the mean Q-value of the Double DQN agent remains consistently below that of the DQN agent. The effect of this consistent overestimation of the mean Q-value by the DQN agent is visible in the cumulative rewards curve (right) as well. The Double DQN agent vastly outperforms the DQN agent here.

## What is the verdict?

The [paper on Double DQN](https://arxiv.org/pdf/1509.06461.pdf) sees the authors performing their experiments on the [Atari environment](https://gym.openai.com/envs/#atari). In the paper, the authors conclude that *"DQN is consistently and sometimes vastly overoptimistic about the value of the current greedy policy"*. They found that, for a majority of the cases, Double DQN displayed better results. The plots below, taken from the paper, support this claim. The graphs present the mean Q-value per step (top row) and cumulative reward per step (bottom) of the two agents. These graphs show Double DQN consistently outperforming DQN.

<img src="/images/posts/20210411_double_dqn/results_double_dqn_paper.png" class="large" alt="">
<em>Fig 5: A screenshot from the paper on Double DQN comparing the mean Q-values (top row) and the corresponding cumulative rewards (bottom row) of the DQN and Double DQN on Atari environments. The top graph shows the Double DQN agent consistently estimating Q-value much lower than DQN, which corresponds to higher scores in the graph in the bottom row.</em>


However, based on our experiments in this post, it is safe to say there are no such performance guarantees. The performance of the Double DQN agent (against DQN) varies with the environment and the hyperparameters. [This StackExchange answer](https://ai.stackexchange.com/questions/11401/what-are-the-differences-between-the-dqn-variants) confirms our finding stating that Double DQN does not really help that much.

## See ya

In this post, we looked at overestimation bias and how it motivated innovation by way of Double DQN. We saw how to update our code to go from DQN to Double DQN. We then studied how much of an effect this update on DQN really has. We also used graphs to visualise how overestimation bias affects the performance of an agent. We finally concluded that, contrary to the Double DQN paper, Double DQN is not necessarily always better than DQN.

Hope you found the post helpful. I'd like to hear from you. Have you worked with Double DQN? Did you find Double DQN better for your task? Please write to me about your experience with Double DQN via email at [saasha.allthingsai@gmail.com](mailto:saasha.allthingsai@gmail.com) or via [Twitter](https://twitter.com/saashanair).

It was nice hanging out with you. Hope to see you again in the next post.

Until then. Ciao! 👋

## Lingering thoughts

1. What was the cause of the constant overestimation in DQN as noticed by the authors? Was it related to the convolutional layers in the network that controlled the agent? Or was it associated with the complexity of the Atari environment as compared against the envrionments studied in this post?

---

## Appendix: Hyperparameters used for the experiments

- **Network architecture:** Dense network with two hidden layers of 400 and 300 units, respectively.
- **Environment:** LunarLander-v2
- **Optimiser:** Adam with a learning rate of 1e-3
- **Discount factor:** 0.99
- **Memory capacity:** 10,000
- **Update frequency:** 10, 100 and 1000
- **Loss function:** Mean Squared Error (MSE)
- **Epsilon range for training:** From max at 1.0 to min at 0.01
- **Epsilon decay for training:** 0.995 with exponential decay applied at each episode
- **Epsilon value for testing:** Fixed at 0.0
- **Random seeds for training:** 12321
- **Random seed for testing:** [456, 12, 985234, 123, 3202]

## Further reading

1. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf), Hado van Hasselt et. al. -- the paper on Double DQN
2. [Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf), Volodymyr Mnih et. al. -- the 2015 Nature-variant of DQN
3. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf), Volodymyr Mnih et. al. -- the 2013 vanilla-DQN (uses only Replay Memory and no Target Network)
4. [DQN Zoo: Reference implementations of DQN-based agents](https://github.com/deepmind/dqn_zoo), John Quan and Georg Ostrovski -- DeepMind's JAX-based implementation of the variants of DQN