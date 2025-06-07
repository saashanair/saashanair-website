---
title: "From algorithm to code: DDPG"
date: 2021-02-15
tags: "Reinforcement Learning"
math: true
---

**NOTE 1**: This is the second post in a two-part series on DDPG.

- {{< backlink "20210201_ddpg_theory" "Part 1">}}: What are the components of the algorithm and how do they work?
- Part 2: How to translate the algorithm to code?

**NOTE 2**: The code relating to this series can be found [on GitHub](https://github.com/saashanair/rl-series/tree/master/ddpg).

---

{{< toc >}}

---

Hiya, how are you doing? First of all, sorry for missing a post last week, got caught up with work. But hopefully, that gave you enough time to dive deeper into DDPG and attempt to implement it yourself. If not, let's walk through the code together. 🤗

## Recap: DDPG Theory 🔁

<img src="/images/posts/20210215_ddpg_algo_to_code/ddpg-algo.png" class="large" alt="">
<em>Fig 1: DDPG Algorithm</em>

In the {{< backlink "20210201_ddpg_theory" "previous post">}}, we saw that DDPG is an off-policy deterministic actor-critic algorithm that combines the principles of DQN and Deterministic Policy Gradient (DPG) to learn to act in a continuous action space. It uses 4 networks - one actor, one critic and their corresponding targets. The critic computes a Q-value, similar to DQN to determine how good the proposed action is in the given state, while the actor predicts the action that the agent should execute.

## Code structure

If you have already [implemented TD3 before](https://www.saashanair.com/td3-code/), you would notice quite a few similarities (which means a major portion of the code is reusable 🤪). The next few paragraphs highlight the nuances discussed in the [DDPG paper](https://arxiv.org/abs/1509.02971).

### Replay Buffer

Being a fairly standard component used in multiple algorithms, such as DQN, TD3 and so on, let's tackle this one first. This class represents a finite-sized buffer with two functionalities:

- the ability to store new experiences as the agent interacts with the environment
- the ability to return a batch of experiences sampled uniformly at random for the agent to learn

There are of course multiple ways for the buffer to be implemented, shown below is just one of them.

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

### Actor

This network accepts the observation obtained from the environment as input and outputs the action to be performed. Thus, the input layer has the same size as the state vector, and the size of the output layer is the same as the action vector. The network also uses two hidden layers, one with 400 units and the other with 300.

<img src="/images/posts/20210215_ddpg_algo_to_code/actor-net.png" class="large" alt="">
<em>Fig 2: Architecture of the actor network</em>


The paper additionally suggests the following:

- **initialisation of layers:** The weights and biases of the final layer were initialised from a uniform distribution of $[-3 \times 10^-3, +3 \times 10^-3]$, while the other layers are initialised from a uniform distribution of $[-\frac{1}{\sqrt{f}}, +\frac{1}{\sqrt{f}}]$, where $f$ is the fan-in (i.e., number of inputs) of the layer.
    
    ```python
    def get_fan_in_init_bound(layer):
        """
        Function to compute the initialisation bound at 1/sqrt(f), where f is the fan-in value (i.e., number of inputs) of the given layer
    
        Parameters
        ---
        layer: torch.nn.module
            The layer of the network to be initialised
    
        Returns
        ---
        the fan-in based upper bound to be used for initialisation, such that the lower bound is the negative of this value
        """
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor.weight)
        #fan_in = layer.weight.size(1) ## a potential solution to computing fan-in when using linear layers as the shape of the weight of the linear layer is [fan_out, fan_in]
        return 1/math.sqrt(fan_in)
    
    def apply_uniform_init(layer, bound=None):
        """
        Function to initialise the specified layer using either the provided bound value or the fan-in based bound (suggested in the DDPG paper for hidden layers)
    
        Parameters
        ---
        layer: torch.nn.module
            The layer of the network to be initialised
    
        bound: float or None
            Specifies the value for the upper bound of the initialisation, such that the lower bound is the negative of this value. If None, then use fan-in based initilisation
    
        Returns
        ---
        none
        """
        if bound is None:
            bound = get_fan_in_init_bound(layer)
        nn.init.uniform_(layer.weight, a=-bound, b=bound) # initalise the weights
        nn.init.uniform_(layer.bias, a=-bound, b=bound) # initialise the biases
    ```
    

- **batch normalisation:** The ranges of the features that form the state vector can vary quite widely. Using batch normalisation can help scale these features appropriately. Thus, in the actor class, batch normalisation is applied after each layer, except for the final output layer.

```python
class Actor(nn.Module):
    """
    Class that defines the neural network architecture for the Actor
    """

    def __init__(self, state_dim, action_dim, max_action, lr=1e-4):
        super(Actor, self).__init__()

        self.max_action = max_action

        self.dense1 = nn.Linear(state_dim, 400)
        apply_uniform_init(self.dense1)

        self.bn1 = nn.BatchNorm1d(400)

        self.dense2 = nn.Linear(400, 300)
        apply_uniform_init(self.dense2)

        self.bn2 = nn.BatchNorm1d(300)

        self.dense3 = nn.Linear(300, action_dim)
        apply_uniform_init(self.dense3, bound=3*10e-3)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = F.relu(self.bn1(self.dense1(state)))
        x = F.relu(self.bn2(self.dense2(x)))
        x = torch.tanh(self.dense3(x)) ## squashes the action output to a range of -1 to +1

        return  self.max_action * x ## assumes action range is symmetric

    ...
```

### Critic

This network, too, uses two hidden layers, with 400 and 300 units respectively. Since this network is tasked with predicting the Q-values for each state-action pair, the output layer is composed of only a single node. As for the input, the network expects to receive information about the state vector and the action vector. The paper suggests passing the two vectors in stages. Initially the network performs a certain amount of computation on the state vector, then this intermediate result is concatenated with the action vector for further processing.

<img src="/images/posts/20210215_ddpg_algo_to_code/critic-net.png" class="large" alt="">
<em>Fig 3: Architecture of the critic network</em>


Similar to the actor, the network for  the critic applies the following:

- **Initialisation of the layers:** Each of the layers use a fan-in based range, while the final output layer is initialised in the range of $[-3 \times 10^{-3}, +3 \times 10^{-3}]$.
- **Batch normalisation:** Since this is used to scale the features of the state vector, it is applied only to the layers prior to the action input.
- **L2 Weight decay:** Applied on to the critic network with a factor of $10^{-2}$ to control overfitting.

```python
class Critic(nn.Module):
    """
    Class that defines the neural network architecture for the Critic
    """

    def __init__(self, state_dim, action_dim, lr=1e-3):
        super(Critic, self).__init__()

        self.dense1 = nn.Linear(state_dim, 400) ## the input to the network is a concatenation of the state and the action performed by the agent in that state
        apply_uniform_init(self.dense1)

        self.bn1 = nn.BatchNorm1d(400)

        self.dense2 = nn.Linear(400 + action_dim, 300)
        apply_uniform_init(self.dense2)

        self.dense3 = nn.Linear(300, 1)
        apply_uniform_init(self.dense3, bound=3*10e-4)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-2)

    def forward(self, state, action):
        x = F.relu(self.bn1(self.dense1(state)))

        x = torch.cat([x, action], dim=1)

        x = F.relu(self.dense2(x))
        x = self.dense3(x) ## the computed Q-value for the given state-action pair

        return x
```

### Ornstein-Uhlenbeck (OU) Noise

With the DDPG agent learning a deterministic policy, we need a technique to encourage exploration. This is often done by adding a small amount of noise to the actions during training. The choice of noise depends largely on the environment being used. For their particular case of robotic control problems with inertia, the authors of the DDPG paper use a time-correlated noise generated via an Ornstein-Uhlenbeck (OU) process.

<img src="/images/posts/20210215_ddpg_algo_to_code/ll-ou.png" class="large" alt="">
<em>Fig 4: The random value generated by the OU Noise for the left and main engines in the LunarLanderContinuous-v2 envinronment over 1000 iterations. The OU Noise was applied with sigma 0.2, theta 0.15 and mean 0.0 (indicated by the black dotted line).</em>


"But what is Ornstein-Uhlenbeck?", you might ask. *OU is a mean-reverting process.* In other words, it simulates a random walk in the presence of a frictional force that continuously acts to push the random values being generated back towards the mean (as shown in Fig 4 above). The first 15-ish minutes of [this video](https://www.youtube.com/watch?v=RFj8SPuZ43Y&t=710s) (despite the overuse of 'like' and 'okay' and the incorrect pronunciation of the name of the process 😅) is the best, most intuitive explanation I have found on the topic, though it does use financial modeling as the context.

The implementation of this process uses three parameters: $\mu$ is the mean, $\theta$ is the frictional force to be applied for the mean-reverting behaviour and $\sigma$ is the amount of noise to be applied. As per [this StackExchange post](https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab) (which is also the basis of [OpenAI's implementation of the OU noise](https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py)), the formula for the OU process can be represented as:

$X_{n+1} = X_{n} + \theta \cdot (\mu - X_{n}) \cdot dt + \sigma \cdot \sqrt{dt} \cdot \mathcal{N}(0, 1)$

```python
class OrnsteinUhlenbeckNoise():
    """
    Class for the OU Process used for generating noise to encourage the agent to explore the environment
    """
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x_start=None):
        self.mu = mu # mean value around which the random values are generated
        self.sigma = sigma # amount of noise to be applied to the process
        self.theta = theta # amount of frictional force to be applied
        self.dt = dt
        self.x_start = x_start # the point from where the random walk is started

        self.reset()

    def reset(self):
        """
        Function to revert the OU process back to default settings. If x_start is specified, use it, else, start from zero.

        Parameters
        ---
        none

        Returns
        ---
        none
        """
        self.prev_x = x_start if self.x_start is not None else np.zeros_like(self.mu)

    def generate_noise(self):
        """
        Function to generate the next value in the random walk which is then used a noise added to the action during training to encourage exploration.
        Formula:
            X_next = X_prev + theta * (mu - X_prev) * dt + sigma * sqrt(dt) * n, where 'n' is a random number sampled from a normal distribution with mean 0 and standard deviation 1

        Parameters
        ---
        none

        Returns
        ---
        none
        """
        x = self.prev_x + self.theta * (self.mu - self.prev_x) * self.dt + \\
                self.sigma * np.sqrt(self.dt) * np.random.normal(loc=0.0, scale=1.0, size=self.mu.shape)

        self.prev_x = x
        return x
```

"Is this absolutely necessary for DDPG?", you might wonder. And the answer to that is, "NO!". In fact, papers discussing the later iterations of the algorithm, such as [TD3](https://arxiv.org/pdf/1802.09477.pdf) and [D4PG](https://arxiv.org/pdf/1804.08617.pdf), found that *in most cases, using a small Gaussian noise is sufficient*.

### DDPGAgent

We are finally at the fun bit. Now we need to put together all the classes we have created thus far. This class encapsulates the logic for:

- how the 4 networks, the replay buffer and the noise generating process are instantiated
- how the action is selected for the given state
- how the networks learn based on agent-environment interactions
- how the target networks are updated

```python
class DDPGAgent:
    """
    Encapsulates the functioning of the DDPG agent
    """

    def __init__(self, state_dim, action_dim, max_action, device, memory_capacity=10000, discount=0.99, tau=0.005, sigma=0.2, theta=0.15, actor_lr=1e-4, critic_lr=1e-3, train_mode=True):
        self.train_mode = train_mode # whether the agent is in training or testing mode

        self.state_dim = state_dim # dimension of the state space
        self.action_dim = action_dim # dimension of the action space

        self.device = device # defines which cuda or cpu device is to be used to run the networks
        self.discount = discount # denoted a gamma in the equation for computation of the Q-value
        self.tau = tau # defines the factor used for Polyak averaging (i.e., soft updating of the target networks)
        self.max_action = max_action # the max value of the range in the action space (assumes a symmetric range in the action space)

        # create an instance of the replay buffer
        self.memory = ReplayMemory(memory_capacity)

        # create an instance of the noise generating process
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(self.action_dim), sigma=sigma, theta=theta)

        # instances of the networks for the actor and the critic
        self.actor = Actor(state_dim, action_dim, max_action, actor_lr)
        self.critic = Critic(state_dim, action_dim, critic_lr)

        # instance of the target networks for the actor and the critic
        self.target_actor = Actor(state_dim, action_dim, max_action, actor_lr)
        self.target_critic = Critic(state_dim, action_dim, critic_lr)

        # initialise the targets to the same weight as their corresponding current networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # since we do not learn/train on the target networks
        self.target_actor.eval()
        self.target_critic.eval()

        # for test mode
        if not self.train_mode:
            self.actor.eval()
            self.critic.eval()
            self.ounoise = None

        self.actor.to(self.device)
        self.critic.to(self.device)

        self.target_actor.to(self.device)
        self.target_critic.to(self.device)

    def select_action(self, state):
        """
        Function to return the appropriate action for the given state.
        During training, it adds a zero-mean OU noise to the action to encourage exploration.
        No noise is added to the action decision during testing mode.

        Parameters
        ---
        state: vector or tensor
            The current state of the environment as observed by the agent

        Returns
        ---
        A numpy array representing the noisy action to be performed by the agent in the current state
        """

        if not torch.is_tensor(state):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)

        self.actor.eval()
        act = self.actor(state).cpu().data.numpy().flatten() # performs inference using the actor based on the current state as the input and returns the corresponding np array
        self.actor.train()

        noise = 0.0

        ## for adding Gaussian noise (to use, update the code pass the exploration noise as input)
        #if self.train_mode:
        #	noise = np.random.normal(0.0, exploration_noise, size=act.shape) # generate the zero-mean gaussian noise with standard deviation determined by exploration_noise

        # for adding OU noise
        if self.train_mode:
            noise = self.ou_noise.generate_noise()

        noisy_action = act + noise
        noisy_action = noisy_action.clip(min=-self.max_action, max=self.max_action) # to ensure that the noisy action being returned is within the limit of "legal" actions afforded to the agent; assumes action range is symmetric

        return noisy_action

    def learn(self, batchsize):
        """
        Function to perform the updates on the 4 neural networks that run the DDPG algorithm.

        Parameters
        ---
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

        with torch.no_grad():
            # generate target actions
            target_action = self.target_actor(next_states)

            # calculate TD-Target
            target_q = self.target_critic(next_states, target_action)
            target_q[dones] = 0.0 # being in a terminal state implies there are no more future states that the agent would encounter in the given episode and so set the associated Q-value to 0
            y = rewards + self.discount * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, y).mean()

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # actor loss is calculated by a gradient ascent along the crtic, thus need to apply the negative sign to convert to a gradient descent
        pred_current_actions = self.actor(states)
        pred_current_q = self.critic(states, pred_current_actions)
        actor_loss = - pred_current_q.mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # apply slow-update to the target networks
        self.soft_update_targets()

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
        self.soft_update_net(self.critic.parameters(), self.target_critic.parameters())

    ...
```

## The final part: main.py

Awesome! We now have a minimal working implementation (does not show the code for saving, loading and so on) of the entire agent. The final bit is to write the training and testing loops so that the agent can learn through interactions with the environment.

The training loop is composed of observing the state of the environment, selecting a noisy action to be performed based on the agent's prediction, executing suggested action, storing the experience in memory, calling the learn function of the agent and repeat on and on!

```python

    # train loop

    ...

    fill_memory(env, ddpg_agent, epochs_fill_memory) # to populate the replay buffer before learning begins
    print('Memory filled: ', len(ddpg_agent.memory))

    for ep_cnt in range(epochs_train):
        done = False
        state = env.reset()
        ep_reward = 0

        while not done:
            action = ddpg_agent.select_action(state) # generate noisy action
            next_state, reward, done, _ = env.step(action) # execute the action in the environment
            ddpg_agent.memory.store([state, action, next_state, reward, done]) # store the interaction in the replay buffer

            ddpg_agent.learn(batchsize=batchsize) # update the networks

            state = next_state

        print('Ep: {} | Ep reward: {} | Moving avg: {}'.format(ep_cnt, ep_reward, moving_avg_reward))

    ...
```

The test loop, on the other hand, requires you to select a non-noisy action to be executed and does not need to store the experiences to memory or call the learn function.

```

    # test loop

    ...

    for ep_cnt in range(epochs_test):
        state = env.reset()
        done = False

        ep_reward = 0
        while not done:
            action = ddpg_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += 1
        print('Ep: {} | Ep reward: {}'.format(ep_cnt, ep_reward))

    ...

```

## Results

<img src="/images/posts/20210215_ddpg_algo_to_code/train-vs-test.png" class="large" alt="">
<em>Fig 5: Train vs Test plot of the DDPG algorithm in the LunarLanderContinuous-v2 environment</em>


And there we have it, a working implementation of the DDPG algorithm! As the graph shows, the model learns to perform in the LunarLanderContinuous environment. The agent was trained without batch normalisation using a learning rate of 1e-3 and 1e-4 for the critic and actor respectively. Additionally, if you recall [the graph from the TD3 post](https://www.saashanair.com/td3-code/), you would notice that DDPG performs a lot worse (hence the need for the [improvements introduced by TD3](https://www.saashanair.com/td3-theory/) 🤪).

## So what's next?

As always, do try to implement the code on your own. But if you need assistance at any point, you can [find the entire the code on GitHub](https://github.com/saashanair/rl-series/tree/master/ddpg). You can, also, reach out to me via [saasha.allthingsai@gmail.com](mailto:saasha.allthingsai@gmail.com) or via [Twitter](https://twitter.com/saasha_nair). 💌

Thank you, dear reader, for sticking around till the end. Hope to see you again in the final instalment in the series, where we would look at the effects of various hyperparameters, especially all the quirky nuances that the paper suggests.

See you soon! 🤓

---

## Further reading

1. [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf), Timothy Lillicrap et. al. -- the paper on DDPG
2. [OU Noise Implementation](https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py), OpenAI Baselines -- code for the Ornstein-Uhlenbeck process
3. [Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html), by OpenAI Spinning -- breaks down the various parts of the algorithm to support quick and easy implementaiton
4. [Implementing DDPG algorithm on the Inverted Pendulum Problem](https://keras.io/examples/rl/ddpg_pendulum/), by [Keras.io](http://keras.io/) -- a keras-based implementation of the DDPG algorithm