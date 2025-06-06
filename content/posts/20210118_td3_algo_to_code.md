---
title: "From algorithm to code: TD3"
date: 2021-01-18
tags: "Reinforcement Learning"
math: true
---

**NOTE 1**: This is the second post in a three-part series on the TD3 algorithm.

- {{< backlink "20210112_td3_theory" "Part 1">}}: theory explaining the different components that build up the algorithm.
- Part 2: how the algorithm is translated to code.
- [Part 3](https://www.saashanair.com/td3-hyperparameter-experiments/): how different hyperparameters affect the behaviour of the algorithm.

**NOTE 2:** The complete code discussed in this post can be found [here](https://github.com/saashanair/rl-series/tree/master/td3) on Github.

---

{{< toc >}}

---

Hello there, dear reader! Welcome back to the next instalment on the TD3 algorithm. I hope you are doing well! Did you attempt to implement the code for the algorithm based on the theory that we discussed in the {{< backlink "20210112_td3_theory" "previous post">}}? If yes, give yourself a big pat on the back. But in case you are still stuck with the implementation, let's work through the code together! 🤓

## Recap: TD3 Theory 🔁

The success of TD3 can be attributed to three improvements over the DDPG algorithm, namely:

1. Clipped Double Q-learning
2. Delayed policy and target updates
3. Target policy smoothing

Compared to DDPG, TD3 also does away with the explicit initialisation of layers of the Actor- and Critic-networks and replaces the Ornstein-Uhlenbeck noise with a small zero-mean Gaussian noise for exploration.

<img src="/images/posts/20210118_td3_algo_to_code/algo.png" class="large" alt="">
<em>Fig 1: The TD3 Algorithm</em>


## Structure of the code

Looking at the algorithm a few things jump right out. We need classes that encapsulate the Actor-network, the Critic-network and the Replay Buffer. Additionally, we need a class, let's call it TD3Agent, that encapsulates all the behavioural information, namely, how to sample from the behaviour policy and how to update the critics, the actor and the targets.

### Actor

<img src="/images/posts/20210118_td3_algo_to_code/actor_net.png" class="large" alt="">
<em>Fig 2: Architecture of the neural network for the Actor</em>


Let's use an actor with 2 hidden layers (as shown in Fig. 2 above), with the first one containing 400 neurons and the second one 300. The input to this network would be the state of the environment observed by the agent and the output would be the continuous-valued action to be applied. Thus, for the LunarLanderContinuous environment with an observation space of 8 and an action space of 2, the input layer would have a size of 8, while the output layer would have 2 neurons.

Additionally, for the output from the actor, we need to ensure that the predicted action is within the valid range. To achieve this, the tanh operation is applied to the output layer to squash the results between the range of -1 and +1. This squashed result is then scaled to match the range of valid action values. Do note that this code assumes that all actions lie within the same symmetric range.

```python
class Actor(nn.Module):
    """
    Class that defines the neural network architecture for the Actor
    """

    def __init__(self, state_dim, action_dim, max_action, lr=1e-3):
        super(Actor, self).__init__()

        self.max_action = max_action

        self.dense1 = nn.Linear(state_dim, 400)
        self.dense2 = nn.Linear(400, 300)
        self.dense3 = nn.Linear(300, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = F.relu(self.dense1(state))
        x = F.relu(self.dense2(x))
        x = torch.tanh(self.dense3(x)) ## squashes the action output to a range of -1 to +1

        return  self.max_action * x ## assumes action range is symmetric
```

### Critic

<img src="/images/posts/20210118_td3_algo_to_code/critic_net.png" class="large" alt="">
<em>Fig 3: Architecture of the neural network for the Critic</em>


Let's use a similar 2 hidden layer architecture (as shown in Fig 3 above) with 400 and 300 neurons for the critic as well. The input to this network would be a concatenation of the state of the environment observed by the agent and the action performed in response. The output layer contains only 1 neuron as the critic returns a single Q-value for each state-action pair provided as input.

```python
class Critic(nn.Module):
    """
    Class that defines the neural network architecture for the Critic
    """

    def __init__(self, state_dim, action_dim, lr=1e-3):
        super(Critic, self).__init__()

        self.dense1 = nn.Linear(state_dim + action_dim, 400) ## the input to the network is a concatenation of the state and the action performed by the agent in that state
        self.dense2 = nn.Linear(400, 300)
        self.dense3 = nn.Linear(300, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):

        x = torch.cat([state, action], dim=1)

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x) ## the computed Q-value for the given state-action pair

        return x
```

### Replay Buffer

This one is a standard component. If you have implemented DQN before, you can just use the same code for this class even with TD3. 👻

This class needs to exhibit two main functionalities:

- the ability to store new experiences gathered by the agent's interaction with the environment
- the ability to sample a batch of random experiences from those stored in memory

```python
class ReplayMemory:
    """
    Class representing the replay buffer used for storing experiences for off-policy learning
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [] # create a list of lists, such that each experience added to memory is a list of 5-items of the form (state, action, next_state, reward, done)
        self.idx = 0

    def store(self, transition):
        """
        Function to add the provided transition/experience to the memory, such that transition is a 5-tuple of the form (state, action, next_state, reward, done)

        Parameters
        ---
        transition: list
            List containing 5-elements representing a single interaction of the agent with the environment

        Returns
        ---
        none
        """

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.idx] = transition
        self.idx = (self.idx + 1) % self.capacity # for circular memory

    def sample(self, batchsize, device):
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

        transitions = np.array(random.sample(self.buffer, batchsize))

        states = torch.tensor(transitions[:, 0].tolist(), dtype=torch.float32).to(device)
        actions = torch.tensor(transitions[:, 1].tolist(), dtype=torch.float32).to(device)
        next_states = torch.tensor(transitions[:, 2].tolist(), dtype=torch.float32).to(device)
        rewards = torch.tensor(transitions[:, 3].tolist(), dtype=torch.float32).to(device)
        dones = torch.tensor(transitions[:, 4].tolist()).to(device)

        return states, actions, next_states, rewards, dones
```

There are multiple ways this class can be implemented, e.g. using deques or having separate lists for each of the five elements in the experience tuple and so on. In the code snippet above, I use a single circular list of lists to achieve the desired functionality.

### TD3Agent

This is the most interesting file, as it brings together all the classes we have created till now. This class needs to encapsulate the following functionality:

- creation of the 6 networks
- selection of the action that the agent executes in the environment
- learning from the agent's experiences

Since this is a big class, let's look at the code for each of the functionalities separately.

```python
class TD3Agent:
    """
    Encapsulates the functioning of the TD3 agent
    """

    def __init__(self, state_dim, action_dim, max_action, device, memory_capacity=10000, discount=0.99, update_freq=2, tau=0.005, policy_noise_std=0.2, policy_noise_clip=0.5, actor_lr=1e-3, critic_lr=1e-3, train_mode=True):
        ...

        # create an instance of the replay buffer
        self.memory = ReplayMemory(memory_capacity)

        # instances of the networks for the actor and the two critics
        self.actor = Actor(state_dim, action_dim, max_action, actor_lr)
        self.critic1 = Critic(state_dim, action_dim, critic_lr)
        self.critic2 = Critic(state_dim, action_dim, critic_lr)

        # instance of the target networks for the actor and the two critics
        self.target_actor = Actor(state_dim, action_dim, max_action, actor_lr)
        self.target_critic1 = Critic(state_dim, action_dim, critic_lr)
        self.target_critic2 = Critic(state_dim, action_dim, critic_lr)

        # initialise the targets to the same weight as their corresponding current networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        ...
```

The \_\_*init\_\_* function (as seen in the code snippet above) of the class is used to create state variables for the memory, the actor-network, the two critic-networks and their corresponding target networks. Also, don't forget to initialise the target networks to have the same weights as their corresponding actor- and critic-networks.

```python
class TD3Agent:
    """
    Encapsulates the functioning of the TD3 agent
    """

	...

	def select_action(self, state, exploration_noise=0.1):
        """
        Function to returns the appropriate action for the given state.
        During training, it returns adds a zero-mean gaussian noise with std=exploration_noise to the action to encourage exploration.
        No noise is added to the action decision during testing mode.

        Parameters
        ---
        state: vector or tensor
            The current state of the environment as observed by the agent
        exploration_noise: float, optional
            Standard deviation, i.e. sigma, of the Gaussian noise to be added to the agent's action to encourage exploration

        Returns
        ---
        A numpy array representing the noisy action to be performed by the agent in the current state
        """

        if not torch.is_tensor(state):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)

        act = self.actor(state).cpu().data.numpy().flatten() # performs inference using the actor based on the current state as the input and returns the corresponding np array

        if not self.train_mode:
            exploration_noise = 0.0 # since we do not need noise to be added to the action during testing

        noise = np.random.normal(0.0, exploration_noise, size=act.shape) # generate the zero-mean gaussian noise with standard deviation determined by exploration_noise

        noisy_action = act + noise
        noisy_action = noisy_action.clip(min=-self.max_action, max=self.max_action) # to ensure that the noisy action being returned is within the limit of "legal" actions afforded to the agent; assumes action range is symmetric

        return noisy_action
```

As we saw in the {{< backlink "20210112_td3_theory" "first post in this series">}}, the TD3 agent explores the environment using a stochastic behaviour policy. Thus, we need to add a small Gaussian noise to the action predicted by the actor-network. The perturbed actions then need to be clipped to ensure that they remain within the valid range of values. Do note that this noise is only added during training.

```python
class TD3Agent:
    """
    Encapsulates the functioning of the TD3 agent
    """

	...

	def soft_update_net(self, source_net_params, target_net_params):
        """
        Function to perform Polyak averaging to update the parameters of the provided network

        Parameters
        ---
        source_net_params: list
            trainable parameters of the source, ie. current version of the network
        target_net_params: list
            trainable parameters of the corresponding target network

        Returns
        ---
        none
        """

        for source_param, target_param in zip(source_net_params, target_net_params):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)

    def soft_update_targets(self):
        """
        Function that calls Polyak averaging on all three target networks

        Parameters
        ---
        none

        Returns
        ---
        none
        """

        self.soft_update_net(self.actor.parameters(), self.target_actor.parameters())
        self.soft_update_net(self.critic1.parameters(), self.target_critic1.parameters())
        self.soft_update_net(self.critic2.parameters(), self.target_critic2.parameters())
```

The paper uses Polyak averaging to slowly update the target networks with a $\tau$ value of 0.005.

```python
class TD3Agent:
    """
    Encapsulates the functioning of the TD3 agent
    """

		...

		def learn(self, current_iteration, batchsize):
        """
        Function to perform the updates on the 6 neural networks that run the TD3 algorithm.

        Parameters
        ---
        current_iteration: int
            Total number of steps that have been performed by the agent
        batchsize: int
            Number of experiences to be randomly sampled from the memory for the agent to learn from

        Returns
        ---
        none
        """

        if len(self.memory) < batchsize:
            return
        states, actions, next_states, rewards, dones = self.memory.sample(batchsize, self.device) # a batch of experiences randomly sampled form the memory

        # ensure that the actions and rewards tensors have the appropriate shapes
        actions = actions.view(-1, self.action_dim)
        rewards = rewards.view(-1, 1)

        # generate noisy target actions for target policy smoothing
        pred_action = self.target_actor(next_states)
        noise = torch.zeros_like(pred_action).normal_(0, self.policy_noise_std).to(self.device)
        noise = torch.clamp(noise, min=-self.policy_noise_clip, max=self.policy_noise_clip)
        noisy_pred_action = torch.clamp(pred_action + noise, min=-self.max_action, max=self.max_action)

        # calculate TD-Target using Clipped Double Q-learning
        target_q1 = self.target_critic1(next_states, noisy_pred_action)
        target_q2 = self.target_critic2(next_states, noisy_pred_action)
        target_q = torch.min(target_q1, target_q2).detach() # since we don't need to learn on the targets, we can ignore the gradients (since we are using two losses with the same y value, not using detach will throw an error)
        target_q[dones] = 0.0 # being in a terminal state implies there are no more future states that the agent would encounter in the given episode and so set the associated Q-value to 0
        y = rewards + self.discount * target_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic_loss1 = F.mse_loss(current_q1, y).mean()
        critic_loss2 = F.mse_loss(current_q2, y).mean()

        self.critic1.optimizer.zero_grad()
        critic_loss1.backward()
        self.critic1.optimizer.step()

        self.critic2.optimizer.zero_grad()
        critic_loss2.backward()
        self.critic2.optimizer.step()

        # delayed policy and target updates
        if current_iteration % self.update_freq == 0:

            # actor loss is calculated by a gradient ascent along crtic 1, thus need to apply the negative sign to convert to a gradient descent
            pred_current_actions = self.actor(states)
            pred_current_q1 = self.critic1(states, pred_current_actions)
            actor_loss = - pred_current_q1.mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # apply slow-update to all three target networks
            self.soft_update_targets()
```

We are finally in the learn function, the last and the most involved part of this class. This function contains the logic for updating all 6 neural networks.

We first start by computing the noisy target actions for ***Target Policy Smoothing***. This is done by adding a small Gaussian noise (unrelated to the noise of the behaviour policy) to the action predicted by the target actor-network for the next_state. The noisy target actions and the next_state act as inputs to the two target critic-networks that produce the Q-values, $Q1\'$ and $Q2\'$. In order to apply ***Clipped Double Q-learning***, the minimum of $Q1\'$ and $Q2\'$ is discounted (don't forget to set the Q-values for terminal states to 0.0!) and then summed with the reward values to compute the TD-Target $y$. The loss for each of the critics is computed as the mean-squared-error of the TD-Target $y$  and the output of the (non-target) critic-network.

***Delayed policy and target updates*** are achieved by updating the actor- and target-networks only on alternate steps. The actor update involves performing a gradient ascent along critic1, by computing the Q-value on the state and the associated non-noisy action predicted by the actor-network. To convert the operation to gradient descent, we need to move along the opposite direction, i.e., take the negative of the Q-value obtained. The final step in the 'learn' function involves calling the 'soft_update_targets' function (from earlier) to update all three target-networks with Polyak averaging.

## Bringing it all together: main.py 🍻

Now that we have the code for all the classes that we need, the final step is to build the training loop and testing loop.

```python
...

def fill_memory(env, td3_agent, epochs_fill_memory):
    """
    Function that performs a certain number of epochs of random interactions with the environment to populate the replay buffer

    Parameters
    ---
    env: gym.Env
        Instance of the environment used for training
    td3_agent: TD3Agent
        Agent to be trained
    epochs_fill_memory: int
        Number of epochs of interaction to be performed

    Returns
    ---
    none
    """

    for _ in range(epochs_fill_memory):
        state = env.reset()
        done = False

        while not done:
            action = env.action_space.sample() # do random action for warmup
            next_state, reward, done, _ = env.step(action)
            td3_agent.memory.store([state, action, next_state, reward, done]) # store the transition to memory
            state = next_state

def train(env, td3_agent, epochs_train, epochs_fill_memory, batchsize, exploration_noise, results_folder):
    """
    Function to train the agent

    Parameters
    ---
    env: gym.Env
        Instance of the environment used for training
    td3_agent: TD3Agent
        Agent to be trained
    epochs_train: int
        Number of epochs/episodes of training to be performed
    epochs_fill_memory: int
        Number of epochs/episodes of interaction to be performed
    batchsize: int
        Number of transitions to be sampled from the replay buffer to perform an update
    exploration_noise: float
        Standard deviation, i.e. sigma, of the Gaussian noise applied to the agent to encourage exploration
    results_folder: str
        Location where models and other result files are saved

    Returns
    ---
    none
    """

	  ...

    fill_memory(env, td3_agent, epochs_fill_memory) # to populate the replay buffer before learning begins
    print('Memory filled: ', len(td3_agent.memory))

    total_steps = 0
    for ep_cnt in range(epochs_train):
        done = False
        state = env.reset()
        ep_reward = 0

        while not done:
            action = td3_agent.select_action(state, exploration_noise=exploration_noise) # generate noisy action
            next_state, reward, done, _ = env.step(action) # execute the action in the environment
            td3_agent.memory.store([state, action, next_state, reward, done]) # store the interaction in the replay buffer

            td3_agent.learn(current_iteration=total_steps, batchsize=batchsize) # update the networks

            state = next_state

            ep_reward += reward
            total_steps += 1

    ...
```

The training loop involves using the 'select_action' function from the TD3 class to predict the agent's action in the given state. The agent then executes the action in the environment. The interaction of the agent with the environment is saved as a 5-tuple of \<state, action, next_state, reward, done\> to the replay buffer by calling the 'store' function associated with the class. Finally, the TD3 agent updates its networks using the 'learn' function.

```python
...

def test(env, td3_agent, epochs_test, seed, results_folder):
    """
    Function to test the agent

    Parameters
    ---
    env: gym.Env
        Instance of the environment used for training
    td3_agent: TD3Agent
        Agent to be trained
    epochs_test: int
        Number of epochs/episodes of testing to be performed
    seed: int
        Value of the seed used for testing
    results_folder: str
        Location where models and other result files are saved

    Returns
    ---
    none
    """

    for ep_cnt in range(epochs_test):
        state = env.reset()
        done = False

        ep_reward = 0

        while not done:
            action = td3_agent.select_action(state, exploration_noise=0.0)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += 1
    ...
```

Conversely, during testing, the agent simply needs to perform iterations for the specified number of epochs, calling only the 'select_action' function from the TD3 class, with exploration_noise set to 0.0. The selected action is then performed in the environment, however, there is no need to store the experiences in memory or to perform the learning step.

## Results 📈

Et Voilà! We have successfully converted the TD3 algorithm to a working code in Python! The full code can be found on [Github](https://github.com/saashanair/rl-series/tree/master/td3). The graph below shows the train and test performance of this implementation on OpenAI Gym's LunarLanderContinuous-v2 environment.

<img src="/images/posts/20210118_td3_algo_to_code/ll_critic_onehead.jpeg" class="large" alt="">
<em>Fig 4: Graph showing the train (left) and test (right) performance of the TD3 algorithm. Left: Shows rewards accumulated by the agent per episode (light blue) during training, along with a moving average of the scores of the last 100 episodes (dark blue). Right: Graphs the rewards per episode accumulated by the agent in test mode across 5 seeds. Note: Here a reward of above 200 is considered as acceptably solving the environment (marked by the red dotted line).</em>


## A different way of structuring the code

The authors of the TD3 paper have made [their code publicly available](https://github.com/sfujim/TD3). It is well written and easy to follow. I would suggest spending some time studying it. However, you might notice a few minor differences between their code and what we worked on above. So, let's spend a few minutes discussing these to make the code associated with the paper easier to assimilate. 🤓

```python
class Critic(nn.Module):
    """
    Class that defines the neural network architecture for the Critic.
    Encapsulates two copies of the same network, reperesentative of the two critic outputs Q1 and Q2 described in the paper
    """

    def __init__(self, state_dim, action_dim, lr=1e-3):
        super(Critic, self).__init__()

        # Architecture for Q1
        self.dense1 = nn.Linear(state_dim + action_dim, 400)
        self.dense2 = nn.Linear(400, 300)
        self.dense3 = nn.Linear(300, 1)

        # Architecture for Q2
        self.dense4 = nn.Linear(state_dim + action_dim, 400)
        self.dense5 = nn.Linear(400, 300)
        self.dense6 = nn.Linear(300, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):

        x = torch.cat([state, action], dim=1)

        # Forward pass for Q1
        q1 = F.relu(self.dense1(x))
        q1 = F.relu(self.dense2(q1))
        q1 = self.dense3(q1)

        # Forward pass for Q2
        q2 = F.relu(self.dense4(x))
        q2 = F.relu(self.dense5(q2))
        q2 = self.dense6(q2)

        return q1, q2 # return the Q-values of critic1 and critic2
```

The difference in code can be attributed to the way that the critic is implemented. The critic class in Scott Fujimoto's repository contains two separate neural networks representing the two critics $Q1$ and $Q2$, thus, leading to a two-headed output.

```python
class TD3Agent:
    """
    Encapsulates the functioning of the TD3 agent
    """

    def __init__(self, state_dim, action_dim, max_action, device, memory_capacity=10000, discount=0.99, update_freq=2, tau=0.005, policy_noise_std=0.2, policy_noise_clip=0.5, actor_lr=1e-3, critic_lr=1e-3, train_mode=True):
        ...

        # create an instance of the replay buffer
        self.memory = ReplayMemory(memory_capacity)

        # instances of the networks for the actor and the two critics
        self.actor = Actor(state_dim, action_dim, max_action, actor_lr)
        self.critic = Critic(state_dim, action_dim, critic_lr) # the critic class encapsulates two copies of the neural network for the two critics used in TD3

        # instance of the target networks for the actor and the two critics
        self.target_actor = Actor(state_dim, action_dim, max_action, actor_lr)
        self.target_critic = Critic(state_dim, action_dim, critic_lr)

        # initialise the targets to the same weight as their corresponding current networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

       ...
```

This also means that we need to update the TD3Agent class. With the new implementation, we would need to create only two instances of the critic class (one for current and one for target).

```python
class TD3Agent:
    """
    Encapsulates the functioning of the TD3 agent
    """

		def learn(self, current_iteration, batchsize):
        """
        Function to perform the updates on the 6 neural networks that run the TD3 algorithm.

        Parameters
        ---
        current_iteration: int
            Total number of steps that have been performed by the agent
        batchsize: int
            Number of experiences to be randomly sampled from the memory for the agent to learn from

        Returns
        ---
        none
        """

        ...

        # generate noisy target actions for target policy smoothing
        pred_action = self.target_actor(next_states)
        noise = torch.zeros_like(pred_action).normal_(0, self.policy_noise_std).to(self.device)
        noise = torch.clamp(noise, min=-self.policy_noise_clip, max=self.policy_noise_clip)
        noisy_pred_action = torch.clamp(pred_action + noise, min=-self.max_action, max=self.max_action)

        # calculate TD-Target using Clipped Double Q-learning
        target_q1, target_q2 = self.target_critic(next_states, noisy_pred_action)
        target_q = torch.min(target_q1, target_q2)
        target_q[dones] = 0.0 # being in a terminal state implies there are no more future states that the agent would encounter in the given episode and so set the associated Q-value to 0
        y = rewards + self.discount * target_q

        current_q1, current_q2 = self.critic(states, actions) # the critic class encapsulates two copies of the neural network thereby returning two Q values with each forward pass

        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y) # the losses of the two critics need to be added as there is only one optimiser shared between the two networks
        critic_loss = critic_loss.mean()

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        ...
```

Finally, the portion of the 'learn' function associated with updating the critic-networks would need to be modified. The target Q-values, $Q1\'$ and $Q2\'$ can now be computed with a single forward pass of the target critic-network owing to the two-headed output of the critic class. Similarly, the current Q-values, $Q1$ and $Q2$ are obtained with a single forward pass of the critic network. The losses of the two critics are computed as the mean-squared-error as earlier, with the only difference being that the two loss values are summed together, as both the critic-networks now share a single optimiser.

I am sure you must be wondering, "But Saasha, does the two-headed implementation of the critic class cause a change in performance?" To answer this, let's compare the plots from both the implementations (the initial one discussed earlier, i.e. one-headed critic, and the modified, i.e. two-headed critic).

<img src="/images/posts/20210118_td3_algo_to_code/ll_critic_onevstwohead.jpeg" class="large" alt="">
<em>Fig 5: Graph comparing the performance of the TD3 algorithm with the one-headed (initial implementation) and the two-headed (author inspired implementation) critic on the LunarLanderContinuous-v2 environment. Left: Shows the moving average of the scores of the last 100 episodes. Right: Graphs the rewards per episode accumulated by the agent in test mode across 5 seeds. Note: Here a reward of above 200 is considered as acceptably solving the environment (marked by the red dotted line).</em>


To ensure that the performance of the implementations is comparable, the same hyperparameter settings have been used for both. As per the graph above (Fig 5), the two-headed variant appears to be more stable, but seems to exhibit a lower test time performance than the one-headed version. However, do note that I have not performed rigorous hyperparameter tuning on the implementations.

## So, what's next? 🚀

Dear reader, I now urge you to test the code for yourself and explore the effects of each of the hyperparameters (we would be looking into it in the next post 😉). The complete implementation of both the versions of the code discussed in this post can be found on [Github](https://github.com/saashanair/rl-series/tree/master/td3). The folder 'critic_one_head' refers to the code from the initial walkthrough, while the folder 'critic_two_head' refers to the version based on the implementation provided by Scott Fujimoto, the author of the TD3 paper (available [here](https://github.com/sfujim/TD3)).

Hope you found this post helpful. Dear reader, please do let me know if you like this new format, or if you have any suggestions on how to improve it. You can get in touch with me via  [saasha.allthingsai@gmail.com](mailto:saasha.allthingsai@gmail.com) or via [Twitter](https://twitter.com/saasha_nair).

See you soon and wish you a productive week ahead! 💼

---

## Further reading

1. [TD3 Implementation](https://github.com/sfujim/TD3), by Scott Fujimoto -- code by the author of the paper on TD3
2. [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf), Scott Fujimoto et. al. -- the paper on TD3
3. [Twin Delayed DDPG](https://spinningup.openai.com/en/latest/algorithms/td3.html), by OpenAI Spinning -- breaks down the various parts of the algorithm to support quick and easy implementaiton
4. [TD3: Learning to Run with AI](https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93), by Donal Byrne -- a blogpost that provides a detailed code walkthrough