---
title: "OpenAI Gym: An Introduction"
date: 2020-09-21
tags: "Reinforcement Learning"
---

Reinforcement Learning, built on the premise that an agent learns through its interactions with its environment, presupposes the existence of an environment. [OpenAI Gym](https://gym.openai.com/), thus, attempts to fill this gap, by providing a diverse range of environments within which to test your RL agents.


## Associated Video

{{< youtube cxMuWd83fI8 >}}

## Why OpenAI Gym?

With projects such as [UnityML](https://unity3d.com/machine-learning), [OpenSpiel](https://github.com/deepmind/open_spiel), [ViZDoom](http://vizdoom.cs.put.edu.pl/), [Carla](https://carla.org/) and the like, RL practitioners have a number of environments available to them. Why then is OpenAI Gym so popular?

OpenAI Gym was born out of a need for benchmarks in the growing field of Reinforcement Learning. The sheer diversity in the type of tasks that the environments allow, combined with design decisions focused on making the library easy to use and highly accessible, make it an appealing choice for most RL practitioners. As you would notice over the course of reading this post, Gym makes no assumptions about the structure of the agent, it instead focuses on providing abstractions for the environments. Thus, making it extremely simple to test each agent with multiple environments and also with different algorithms backing the agents.

<img src="/images/posts/20200921_openai_gym_an_introduction/1_openai-gym.png" class="large" alt="A screenshot of OpenAI Gym's landing page describes Gym as a toolkit for developing and comparing reinforcement learning algorithms. It supports teaching agents everything from walking to playing games like Pong or Pinball.">
<em>A succinct description of OpenAI Gym as found on the Landing Page (<a href="https://gym.openai.com"> source</a>)</em>

The environments provided by Gym fall under the following categories:

1. **Classic Control** and **Toy Text**: Simplest environments based on RL literature that are beginner friendly. The 'CartPole' environment, which belongs to this category, can be thought of as the 'MNIST' of RL.
2. **Algorithmic**: The aim here is to learn to imitate computations.
3. **Atari**: Provides an easy to use form of the [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment), which provides access to over 50 Atari games by building over the Atari 2600 emulator.
4. **Box2D**, **MuJoCo** and **Robotics**: This focuses on learning continuous control tasks. 'LunarLander' is the most commonly used environment belonging to this category.

## How to install Gym?

Installation of this toolkit is as simple as using the pip install command.

```bash
pip install gym
```

This command allows the use of environments belonging to Classic Control, Toy Text and Algorithmic categories, but to use an environment such as Breakout from Atari, or LunarLander from Box2D, one is required to perform an extra pip install step.

```bash
# for using Atari environments
pip install gym[atari]

# for using Box2D environments
pip install gym[box2d]
```

To test whether your installations have been completed successfully, ensure that the following section of Python code results in no errors.

```python
import gym
gym.make('CartPole-v0')
gym.make('Breakout-v0') # to test Atari environment
gym.make('LunarLander-v2') # to test Box2D environment
```

## Basic functionality of Gym

Importing Gym into your script using `import gym` allows you to use the following functions.

1. **gym.make(env_id):** is used to create an instance of the target environment
    
    ```python
    env = gym.make('CartPole-v0')
    ```
    
2. **env.reset():** is the first function to be called, before any interactions can be performed between the agent and the environment. It initialises the environment by placing the agent at the start position, and in turn returns the initial observation, i.e the state of the environment.
    
    ```python
    state = env.reset()
    ```
    
3. **env.step(action):** performs the agent action, passed as an argument to the function, within the environment and returns the next state, reward, done and info. Next state contains the observation of the environment that has resulted from the agent performing the action in the environment. The reward acts as a signal to guide the agent towards learning more of the behaviour that accomplishes the desired task, while suppressing the actions that cause the agent to stray away from the goal. Done is a boolean flag that determines whether the terminal state has been reached or not. Info is used to capture additional information from the environment, and is usually an empty dictionary.
    
    ```python
    next_state, reward, done, info = env.step(action)
    ```
    
4. **env.render():** is useful for studying how the agent is interacting with the environment. It renders the environment states as RGB frames, which appear to the human eye as a video depicting agent-environment interactions.
    
    ```python
    env.render()
    ```
    
5. **env.close():** performs the necessary cleanup to close the environment. By default, environments usually close themselves when the program exits, however, using env.render might cause the program to throw out an error message while it exits, and this can be avoided using env.close().
    
    ```python
    env.close()
    ```
    
6. **env.action_space** and **env.observation_space**: are instance variables that define the type and range of the states that the agent can observe and actions that the agent can perform. The two most common types of space are '**Box**', which is used for the continuous domain, and '**Discrete**'.
    
    ```python
    env.action_space
    env.observation_space
    ```
    
    For example, in the CartPole environment, at each step, the agent receives as input an array of real numbers of size 4 as the state information and can perform 2 actions,  as indicated by Box(4,) and Discrete(2) respectively.

    <img src="/images/posts/20200921_openai_gym_an_introduction/2_screenshot.png" class="large" alt="Screenshot showing action space as Discrete(2) and observation space as Box(4,)">
    <em>A screenshot depicting the action and observation spaces of the CartPole environment in OpenAI Gym</em>
    
    A screenshot depicting the action and observation spaces of the CartPole environment in OpenAI Gym
    
    Additionally, you can determine the range of values that action_space and observation_space can take. To determine the range for Discrete space, use the '**n**' instance variable. For the CartPole environment, using the 'n' instance variable with action_space would return '2', which means the agent can take two actions which are represented as 0 and 1. Similarly, for the Box space, as represented by the observation_space in the CartPole example, '**high**' and  '**low**' are used to determine the upper and lower bounds.
    
    ```python
    env.action_space.n
    env.obsevation_space.high
    env.observation_space.low
    ```
    
7. **env.action_space.sample():** allows you to sample an action uniformly at random from all the possible actions available to the agent.
    
    ```python
    action = env.action_space.sample()
    ```
    

## Minimal code for agent-environment interaction

Piecing together the functions discussed above, we can write the following segment of code that will allow us to observe how the functions work to allow an agent to interact with the environment.

```python
import gym

env = gym.make('CartPole-v0')
state = env.reset()

action = env.action_space.sample() # similar to having a random agent, i.e., an agent that performs only random actions and does not learn anything
next_state, reward, done, info = env.step(action)

# to observe the information captured in each of the variables
print('State: '.format(state))
print('Action: '.format(action))
print('Next state: {}; Reward: {}; Done: {}; Info: {}'.format(next_state, reward, done, info))

```

There we have it, we wrote our first piece of code that allows us to work with OpenAI Gym environments. The problem though is that this code allows only one step of interaction with the environment, while in Reinforcement Learning, we require the agent to perform thousands, even millions of steps.

## Skeleton of the code for agent-environment interaction for RL

To support the agent interacting with the environment multiple times, the above code must be modified to use loops. The most common approach is to iterate over the number of episodes, where an episode is the run of the game from the start state to the terminal state. Another approach involves iterating over the maximum time steps of game play, i.e. number of interactions performed. The code below adopts the former method, and can just as easily be modified to iterate over time steps.

```python
import gym

env = gym.make('CartPole-v0')

max_ep = 10 # maximum number of episodes to loop over
for ep_cnt in range(max_ep):
	step_cnt = 0 # tracks number of steps taken in an episode
	ep_reward = 0 # tracks total reward accumulated in an episode
	done = False # boolean flag indicating if terminal state is reached
	state = env.reset() # first observation read from the environment on initialization

	while not done: # loop over agent-environment interactions until terminal state is reached, i.e. episode ends
		next_state, reward, done, _ = env.step(env.action_space.sample()) # perform a random action within the environment
		env.render() # render a frame showing state of environment in a human-friendly format
		step_cnt += 1
		ep_reward += reward
		state = next_state

	print('Episode: {}, Step count: {}, Episode reward: {}'.format(ep_cnt, step_cnt, ep_reward))
env.close()
```

Running the following code, should cause a window to pop up that shows the actions being performed in the cart pole environment.

<img src="/images/posts/20200921_openai_gym_an_introduction/3_cartpole_result.gif" class="large" alt="An episode in the CartPole example created using the above code">
<em>An episode in the CartPole example created using the above code</em>

An episode in the CartPole example created using the above code

Of course, rendering every step of interaction between the agent and the environment might not be a good choice for graphic-heavy environments. In such cases, it might make sense to comment out the env.render() function during the training of the agent, and use it only while testing to observe what the agent has learnt.

**Tip:** If you wish for Gym to save the agent-environment interactions as videos, use `env = gym.wrappers.Monitor(env, '.', force=True)` right after the line containing the `gym.make()` function. When using the Monitor function, ensure that you have ffmpeg installed on your system.

## Conclusion

OpenAI Gym makes it extremely convenient to test your RL agents in different environments. Changing the environment within which the agent must interact is simply a matter of plugging the target environment ID into the gym.make() function. Dear reader, go ahead and replace 'CartPole-v0' with say 'MountainCar-v0' or any other environment (see list of available environments [here](https://gym.openai.com/envs/#algorithmic)), and the code will still work the same. Though there are many environment collections available to test your RL agents, the simplicity of Gym makes it a great first choice, especially while starting out on your RL journey.

Dear reader, thank you for sticking around till the end, hope you found this useful. Looking forward to hearing from you, so please drop me a message at [saasha.allthingsai@gmail.com](mailto:saasha.allthingsai@gmail.com), or hit me up on [Twitter](https://twitter.com/saasha_nair). 💙

See in the next one! 🦾

---

## Resources:

1. [OpenAI Gym Official Website](https://gym.openai.com/)
1. [Official Git Repo of OpenAI Gym](https://github.com/openai/gym)
1. [Official documentation listing the environments available in OpenAI Gym](https://github.com/openai/gym/blob/master/docs/environments.md)
1. [Code describing the structure and methods of the Gym environment class](https://github.com/openai/gym/blob/master/gym/core.py)
1. [A short read describing the design decisions behind OpenAI Gym](https://arxiv.org/pdf/1606.01540.pdf)