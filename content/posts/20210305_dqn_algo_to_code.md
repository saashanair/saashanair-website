---
title: "From algorithm to code: DQN"
date: 2021-03-05
tags: "Reinforcement Learning"
math: true
---

This is the second post in a four-part series on DQN.

- {{< backlink "20210225_dqn_theory" "Part 1">}}: Components of the algorithm
- Part 2: Translating algorithm to code
- {{< backlink "20210321_dqn_and_its_hyperparams" "Part 3">}}: Effects of the various hyperparameters
- {{< backlink "20210411_double_dqn" "Part 4">}}: Combating overestimation with Double DQN

---

{{< toc >}}

---

The {{< backlink "20210225_dqn_theory" "previous post in this series">}} looked at how the DQN algorithm works. Today let's attempt to implement it in PyTorch. Are you pumped? Let's go!

<img src="https://i.giphy.com/media/CjmvTCZf2U3p09Cn0h/giphy.webp" class="large" alt="">
<em><a href="https://media.giphy.com/media/CjmvTCZf2U3p09Cn0h/giphy.gif">Source</a></em>


## Recap: DQN Theory

From {{< backlink "20210225_dqn_theory" "Part 1 of this series">}}, we know that DQN is an off-policy algorithm. It learns to act by computing the Q-value of each possible action in the given state and then picking the action with the highest Q-value. This limits the algorithm to a discrete action space (hence, the need for {{< backlink "20210201_ddpg_theory" "DDPG">}} and {{< backlink "20210112_td3_theory" "TD3">}}!).

<img src="/images/posts/20210305_dqn_algo_to_code/algo.png" class="large" alt="">
<em>Fig 1: The DQN Algorithm</em>


## Code Structure

The schematics of this algorithm are quite simple. We need a neural network to learn the mapping between the state and the Q-value of each of the possible actions, let's call that the DQNNet. The algorithm also uses a Target Network, which in terms of implementation is simply a second instance of the same DQNNet. Additionally, past experiences of the agent are stored in memory using an instance of the class called Replay Buffer.

### Replay Buffer

Replay Buffer is a fixed-length circular memory that overwrites earlier experiences once the memory is full. It needs two main functionalities:

1. *the ability to store new experiences:* This involves keeping track of the available space in the memory to determine where the new experiences are to be placed.
2. *the ability to sample past experiences:* The sampling is performed uniformly at random to obtain a mini-batch of experiences for the agent to learn.

There are multiple ways of implementing this class. The code here uses different arrays for each of the components of the experience, namely, 'state', 'action', 'next_state', 'reward' and 'done'. For an example of another style of implementation, check the corresponding section in the {{< backlink "20210201_ddpg_theory" "DDPG code walkthrough">}}.

```python
 class ReplayMemory:
    """
    Class representing the replay buffer used for storing experiences for off-policy learning
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer_state = []
        self.buffer_action = []
        self.buffer_next_state = []
        self.buffer_reward = []
        self.buffer_done = []
        self.idx = 0

    def store(self, state, action, next_state, reward, done):
        """
        Function to add the provided experience to the memory, such that transition is a 5-tuple of the form (state, action, next_state, reward, done)

        Parameters
        ---
        state: numpy.ndarray
            Current state vector observed in the environment
        action: int
            Action performed by the agent in the current state
        next_state: numpy.ndarray
            State vector observed as a result of performing the action in the current state
        reward: float
            Reward obtained by the agent
        done: bool
            Indicates whether the agent has entered a terminal state or not

        Returns
        ---
        none
        """

        if len(self.buffer_state) < self.capacity:
            self.buffer_state.append(state)
            self.buffer_action.append(action)
            self.buffer_next_state.append(next_state)
            self.buffer_reward.append(reward)
            self.buffer_done.append(done)
        else:
            self.buffer_state[self.idx] = state
            self.buffer_action[self.idx] = action
            self.buffer_next_state[self.idx] = next_state
            self.buffer_reward[self.idx] = reward
            self.buffer_done[self.idx] = done

        self.idx = (self.idx+1)%self.capacity # for circular memory

    def sample(self, batch_size, device):
        """
        Function to pick 'n' samples from the memory that are selected uniformly at random, such that n = batchsize

        Parameters
        ---
        batchsize: int
            Number of elements to randomly sample from the memory in each batch
        device: str
            Name of the device (cuda or cpu) on which the computations would be performed

        Returns
        ---
        Tensors representing a batch of transitions sampled from the memory
        """

        indices_to_sample = random.sample(range(len(self.buffer_state)), batch_size)

        states = torch.from_numpy(np.array(self.buffer_state)[indices_to_sample]).float().to(device)
        actions = torch.from_numpy(np.array(self.buffer_action)[indices_to_sample]).to(device)
        next_states = torch.from_numpy(np.array(self.buffer_next_state)[indices_to_sample]).float().to(device)
        rewards = torch.from_numpy(np.array(self.buffer_reward)[indices_to_sample]).float().to(device)
        dones = torch.from_numpy(np.array(self.buffer_done)[indices_to_sample]).to(device)

        return states, actions, next_states, rewards, dones
```

### DQNNet

<img src="/images/posts/20210305_dqn_algo_to_code/q-net.png" class="large" alt="">
<em>Fig 2: The architecture of the Q-network</em>


Let's model the Q-Network with two hidden layers of 400 and 300 units respectively. The size of the input layer to the network is the same as that of the state vector. Conversely, the size of the output layer matches the number of discrete actions afforded to the agent in the given environment.

```python
class DQNNet(nn.Module):
    """
    Class that defines the architecture of the neural network for the DQN agent
    """
    def __init__(self, input_size, output_size, lr=1e-3):
        super(DQNNet, self).__init__()
        self.dense1 = nn.Linear(input_size, 400)
        self.dense2 = nn.Linear(400, 300)
        self.dense3 = nn.Linear(300, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x
```

### DQNAgent

This is where all the magic happens! ✨

This class is responsible for selecting the actions that the agent performs and also for updating the networks. Let's look at each of these functionalities separately.

First things first, defining all the variables in the __*init__*() function. This involves creating an instance of the Replay Buffer class and two instances of the DQNNet for the Q-network and its target.

```python
class DQNAgent:
    """
    Class that defines the functions required for training the DQN agent
    """
    def __init__(self, device, state_size, action_size,
                    discount=0.99,
                    eps_max=1.0,
                    eps_min=0.01,
                    eps_decay=0.995,
                    memory_capacity=5000,
                    lr=1e-3,
                    train_mode=True):

        self.device = device

        # for epsilon-greedy exploration strategy
        self.epsilon = eps_max
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay

        # for defining how far-sighted or myopic the agent should be
        self.discount = discount

        # size of the state vectors and number of possible actions
        self.state_size = state_size
        self.action_size = action_size

        # instances of the network for current policy and its target
        self.policy_net = DQNNet(self.state_size, self.action_size, lr).to(self.device)
        self.target_net = DQNNet(self.state_size, self.action_size, lr).to(self.device)
        self.target_net.eval() # since no learning is performed on the target net
        if not train_mode:
            self.policy_net.eval()

        # instance of the replay buffer
        self.memory = ReplayMemory(capacity=memory_capacity)
```

The behaviour of the agent changes based on whether it is training or testing. During training, the agent needs to 'explore' the environment. This helps ensure that it does not get stuck in "bad routines" early on. Thus, DQN utilises an {{< backlink "20210225_dqn_theory#exploration-in-dqn" "Annealing Epsilon-Greedy policy">}} during training. Linear or Exponential (used here) are the two most common annealing strategies for adjusting the epsilon value. However, while testing, the epsilon value is set to 0.0 to force the agent to 'exploit' the learnt policy.

```python
	## the following code is placed within the DQNAgent class

	def update_epsilon(self):
        """
        Function for reducing the epsilon value (used for epsilon-greedy exploration with annealing)

        Parameters
        ---
        none

        Returns
        ---
        none
        """

        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

    def select_action(self, state):
        """
        Uses epsilon-greedy exploration such that, if the randomly generated number is less than epsilon then the agent performs random action, else the agent executes the action suggested by the policy Q-network
        """
        """
        Function to return the appropriate action for the given state.
        During training, returns a randomly sampled action or a greedy action (predicted by the policy network), based on the epsilon value.
        During testing, returns action predicted by the policy network

        Parameters
        ---
        state: vector or tensor
            The current state of the environment as observed by the agent

        Returns
        ---
        none
        """

        if random.random() <= self.epsilon: # amount of exploration reduces with the epsilon value
            return random.randrange(self.action_size)

        if not torch.is_tensor(state):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)

        # pick the action with maximum Q-value as per the policy Q-network
        with torch.no_grad():
            action = self.policy_net(state)
        return torch.argmax(action).item() # since actions are discrete, return index that has highest Q
```

Finally, for the agent to learn, the Q-network (referred to as 'policy_net' in the code) is updated by computing the mean-squared-error between the predicted Q-value and the TD-Target (referred to as 'y_j' in the code below).

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
        loss = F.mse_loss(y_j, q_pred).mean()
        loss.backward()
        self.policy_net.optimizer.step()
```

Additionally, we need one last function that would be called at regular intervals to update the Target Network. DQN uses hard updates, meaning after every 'n' steps, the weights of the 'policy_net' are copied into the 'target_net'.

```python
	## the following code is placed within the DQNAgent class

	def update_target_net(self):
        """
        Function to copy the weights of the current policy net into the (frozen) target net

        Parameters
        ---
        none

        Returns
        ---
        none
        """

        self.target_net.load_state_dict(self.policy_net.state_dict())
```

## The final part: main.py

With all the classes and functionality implemented, the last step is to tie it all together with the main.py file that contains the train and test loops.

The train loop involves observing the environment, selecting an action via the epsilon-greedy policy, storing the interaction in the replay buffer, calling the learn function to update the Q-network, and then hard-updating the Target Network at regular intervals.

```python

...

fill_memory(env, dqn_agent, num_memory_fill_eps)
print('Memory filled. Current capacity: ', len(dqn_agent.memory))

step_cnt = 0

for ep_cnt in range(num_train_eps):
	done = False
  state = env.reset()

  while not done:
	  action = dqn_agent.select_action(state)
    next_state, reward, done, info = env.step(action)
	  dqn_agent.memory.store(state=state, action=action, next_state=next_state, reward=reward, done=done)

    dqn_agent.learn(batchsize=batchsize)

    if step_cnt % update_frequency == 0:
	    dqn_agent.update_target_net()

	  state = next_state
    step_cnt += 1

	dqn_agent.update_epsilon()

...
```

The testing loop is even simpler than that. All you need to do here is to interact with the environment by executing the actions returned by the DQNAgent. Don't forget to set the 'epsilon_max' value to 0.0 at this point, as you want to purely exploit the learnt policy here.

```python
...

for ep_cnt in range(epochs_test):
	state = env.reset()
  done = False

  ep_reward = 0
  while not done:
	  action = dqn_agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    ep_reward += 1
  print('Ep: {} | Ep reward: {}'.format(ep_cnt, ep_reward))

    ...
```

## Results

Congrats on your minimal working implementation of DQN! 🎉

You can find the [entire code on Github](https://github.com/saashanair/rl-series/tree/master/dqn) or work through the code on [Binder](https://mybinder.org/v2/gh/saashanair/rl-series/HEAD?filepath=dqn%2Fmain.ipynb).

<img src="/images/posts/20210305_dqn_algo_to_code/dqn-lunarlander.png" class="large" alt="">
<em>Fig 3: Train (left) vs Test (right) plot of the DQN algorithm on the LunarLander environment. The light blue line in the left plot shows the rewards obtained by the agent per episode during training, while the dark blue line is the moving average score obtained over the last 100 episodes. The test time graph shows the average performance (dark blue line) of the agent across 5 seeds, with the light blue bands indicating the variance in the score in each episode. The red dotted line depicts the threshold score for the environment to be declared as completed.</em>


Hyperparameter values used here:

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

## See ya

As always, a big thank you. 🤗

Hope you found the post useful. Oh, and don't forget to say hi on  [Twitter](https://twitter.com/saashanair), or drop me a line at [saasha.allthingsai@gmail.com](mailto:saasha.allthingsai@gmail.com). 💌

Ciao! 👋

---

## Further Reading

1. [Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html), Adam Paszke -- official Pytorch tutorial on DQN that uses images as state information
2. [Implementing Deep Reinforcement Learning Models with Tensorflow + OpenAI Gym](https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html), Lilian Weng -- a detailed tutorial covering Naive Q-Learning, DQN, Double DQN and Dueling DQN
3. [Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf), Volodymyr Mnih et. al. -- the 2015 Nature-variant of DQN
4. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf), Volodymyr Mnih et. al. -- the 2013 vanilla-DQN (uses only Replay Memory and no Target Network)
5. [Let's build a DQN: Basics](https://tomroth.com.au/dqn-basics/), Tom Roth -- a three-part series about how DQN works