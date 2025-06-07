---
title: "DQN and its Hyperparameters"
date: 2021-03-21
tags: "Reinforcement Learning"
math: true
---

This is the third post in a four-part series on DQN.

- {{< backlink "20210225_dqn_theory" "Part 1">}}: Components of the algorithm
- {{< backlink "20210305_dqn_algo_to_code" "Part 2">}}: Translating algorithm to code
- Part 3: Effects of the various hyperparameters
- {{< backlink "20210411_double_dqn" "Part 4">}}: Combating overestimation with Double DQN

---

{{< toc >}}

---

Heyo, how are you doing? Hope you have been having a wonderful week.

In the previous posts of this series on DQN, we have already familiarised ourselves with {{< backlink "20210225_dqn_theory" "the relevant theory">}} and then {{< backlink "20210305_dqn_algo_to_code" "implemented the agent in PyTorch">}}. But to understand any ML model, it is pertinent to experiment with hyperparameters to see how the model responds.

Today we will use the {{< backlink "20210305_dqn_algo_to_code" "DQN implementation from the previous post">}} applied to the [LunarLander environment](https://gym.openai.com/envs/LunarLander-v2/) to look at:

1. How big should the replay memory be?
2. How does the Target Network help?
3. How frequently should the Target Network be updated?
4. How does the choice of loss function affect learning?

Let's jump right into it!

<img src="/images/posts/20210321_dqn_and_its_hyperparams/giddy-up.gif" class="large" alt="">
<em><a href="http://gph.is/1Eo4CCa">Source</a></em>


## Experiment 1: Effect of Memory Capacity

The replay memory tracks the agent's interactions with the environment to provide a pool of data samples from which to learn. If the memory capacity is too small, the agent uses each experience only a few times.

Fig 1 below confirms our hypothesis. With a memory capacity of 1, the agent is barely able to learn. On increasing the memory capacity to 100, the agent appears to start learning, but the performance is still low. The agent with a memory capacity of 10,000 crosses the threshold score and beats the environment.

<img src="/images/posts/20210321_dqn_and_its_hyperparams/memory-capacity.png" class="large" alt="">
<em>Fig 1: Train (left) vs Test (right) plot showing the effect of varying the capacity of the replay buffer</em>


Changing the memory capacity also affects training speed, despite the batch size remaining unchanged at 64. As the memory capacity goes up, so does the time to complete 2000 training episodes. The experiment with the memory capacity set to 100,000 ran for 3 days before the kernel died but reached only 1400 training episodes, thus not making it to this plot. 🤦‍♀️

## Experiment 2: Effect of Target Network

Target Networks were added to DQN in the 2015 Nature variant of the algorithm. As we had established in {{< backlink "20210225_dqn_theory#target-network" "Part 1 of this series">}}, the Target Network helps stabilise learning by reducing divergent behaviour.

Let's test the efficacy of the Target Network. Update the 'learn' function of the 'DQNAgent' class (from the previous post) to compute the TD-Target via the Policy Network. It requires a single line of change, as shown in the code snippet below.

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

        # get q values of the actions that were taken, i.e calculate qpred;
        # actions vector has to be explicitly reshaped to nx1-vector
        q_pred = self.policy_net(states).gather(1, actions.view(-1, 1))

        #calculate target q-values, such that yj = rj + q(s', a'), but if current state is a terminal state, then yj = rj
        #q_target = self.target_net(next_states).max(dim=1).values # because max returns data structure with values and indices

		## CHANGE: instead of using target_net (as above), q_target is computed using policy_net (as below)
		q_target = self.policy_net(next_states).max(dim=1).values
        ## END OF CHANGE

        q_target[dones] = 0.0 # setting Q(s',a') to 0 when the current state is a terminal state
        y_j = rewards + (self.discount * q_target)
        y_j = y_j.view(-1, 1)

        # calculate the loss as the mean-squared error of yj and qpred
        self.policy_net.optimizer.zero_grad()
        loss = F.mse_loss(q_pred, y_j).mean()
        loss.backward()
        self.policy_net.optimizer.step()
```

Comparing the performance of an agent with and without a Target Network, the benefits are clear. Despite the two agents being trained with the same hyperparameters, the agent without a Target Network exhibits noisy test time behaviour. This high variance is noticed as a wide band around the mean score (dark red line) in Fig 2 below. In comparison, the agent with a Target Network shows a tight band around the mean score (dark blue line), with most of the episodes scoring above the threshold.

<img src="/images/posts/20210321_dqn_and_its_hyperparams/with-vs-without-target-net.png" class="large" alt="">
<em>Fig 2: Train (left) vs Test (right) plot showing the efficacy of using a Target Network.</em>


## Experiment 3: Update Frequency of the Target Network

Alright, so we agree on the importance of Target Network. The next logical question then is, how frequently should it be updated?

If the Target Network is updated too frequently, the TD-Target will be unstable, causing divergent behaviour. Conversely, if the Target Network is not updated often enough, the TD-Target would lead the agent down the wrong path.

<img src="/images/posts/20210321_dqn_and_its_hyperparams/update-frequency.png" class="large" alt="">
<em>Fig 3: Train (left) vs Test (right) time performance resulting from varying the frequency at which the Target Network is updated.</em>


Fig 3 above is consistent with our hypothesis. An update frequency of 1 and 10,000 are damaging to the agent's performance. The agent appears to perform well in the LunarLander-v2 environment when the update frequency is set to either 10 or 1000. However, I found it baffling that the update frequency of 100 did not perform well during testing. If you have any clues of why that might be, please do let me know. 😅

## Experiment 4: Effect of Loss Functions

Mean Squared Error or MSE is one of the most commonly used loss functions. As the name suggests, it is the sum of the squared distance between the TD-Target and the predicted Q value. The squaring operation makes MSE stringent, causing the incurred penalty to increase with the error. This means when the difference between the TD-Target and the predicted Q-value is high, the agent receives a higher penalty or "punishment". This can be problematic in the RL settings where the TD-Target is only an estimate and not the ground truth. In other words, we would prefer to take the TD-Target estimates with a grain of salt and treat it as a guiding force instead of it being the absolute truth. An alternative loss function that is more forgiving than MSE is [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss).

Huber Loss can be seen as a combination of MAE (Mean Average Error) and MSE. When the error is small, it applies a quadratic penalty, similar to MSE. But when the error is large, a linear penalty is applied, similar to MAE. The definition of "large" and "small" are controlled in the loss function via a hyperparameter.

The experiment requires replacing the loss function in the 'DQNAgent' class (from the previous post). PyTorch implements Huber loss under the name '[smooth_l1_loss](https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html)' (check [PyTorch's Official Tutorial on DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)). Thus, we required only a single line of change within the 'learn' function.

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

        # get q values of the actions that were taken, i.e calculate qpred;
        # actions vector has to be explicitly reshaped to nx1-vector
        q_pred = self.policy_net(states).gather(1, actions.view(-1, 1))

        #calculate target q-values, such that yj = rj + q(s', a'), but if current state is a terminal state, then yj = rj
        q_target = self.target_net(next_states).max(dim=1).values # because max returns data structure with values and indices
        q_target[dones] = 0.0 # setting Q(s',a') to 0 when the current state is a terminal state
        y_j = rewards + (self.discount * q_target)
        y_j = y_j.view(-1, 1)

        # calculate the loss as the mean-squared error of yj and qpred
        self.policy_net.optimizer.zero_grad()
        #loss = F.mse_loss(q_pred, y_j).mean()
		## CHANGE: use smooth_l1_loss instead
		loss = F.smooth_l1_loss(q_pred, y_j).mean()
        ## END OF CHANGE
        loss.backward()
        self.policy_net.optimizer.step()
```

This experiment did not fare too well. Given the simplicity of the LunarLander environment, I had expected Huber Loss to perform as well as MSE Loss, if not better. However, with all other hyperparameters kept equal, Huber Loss exhibited noisy test time performance, as seen in Fig 4 below.

<img src="/images/posts/20210321_dqn_and_its_hyperparams/huber-vs-mse-loss.png" class="large" alt="">
<em>Fig 4: Train (left) vs Test (right) plot showing the effect of varying the loss function.</em>


A possible explanation for the poor performance of Huber Loss in this experiment could be that the hyperparameters were optimised for MSE. I hypothesise that performance should improve with hyperparameter tuning. If you do give it a try, please let me know how your experiment turns out. 😁

## See ya

Hope this post added to your understanding of the DQN agent and inspired you to experiment with it. I would love to hear about your experience with DQN. Please do drop me a line via email at [saasha.allthingsai@gmail.com](mailto:saasha.allthingsai@gmail.com) or on [Twitter](https://twitter.com/saashanair). 💌

See you in the next post! 👋

---

## Appendix: Default Hyperparameters used for the experiment

- **Network architecture:** Dense network with two hidden layers of 400 and 300 units, respectively.
- **Environment:** LunarLander-v2
- **Optimiser:** Adam with a learning rate of 1e-3
- **Discount factor:** 0.99
- **Memory capacity:** 10,000
- **Update frequency:** 1000
- **Loss function:** Mean Squared Error (MSE)
- **Epsilon range for training:** From max at 1.0 to min at 0.01
- **Epsilon decay for training:** 0.995 with exponential decay applied at each episode
- **Epsilon value for testing:** Fixed at 0.0
- **Random seeds for training:** 12321
- **Random seed for testing:** [456, 12, 985234, 123, 3202]

## Further reading

1. [Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html), by Adam Paszke -- official Pytorch tutorial on DQN that shows how to use Huber Loss
2. [Section 6.1 of Grokking Deep Reinforcement Learning (Manning Early Access Program Edition)](https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-6/v-4/61), by Miguel Morales -- an in-depth discussion on the MSE, MAE and Huber Loss explaining why MSE might not always be the best choice