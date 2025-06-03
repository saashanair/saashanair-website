---
title: "OpenAI Gym and Custom Environments"
date: 2020-09-26
tags: "Reinforcement Learning"
---

Consider this situation. You are tasked with training a Reinforcement Learning Agent that is to learn to drive in [The Open Racing Car Simulator (TORCS)](https://en.wikipedia.org/wiki/TORCS). However, instead of diving into a complex environment, you decide to build and test your RL Agent in a simple Gym environment to hammer out possible errors before applying hyperparameters tuning to port the agent to TORCS. The code for the agent, however, when ported to TORCS does not work because the methods/functions used to interact with the TORCS environment are different from those used in Gym. Hence you are required to completely refactor your code. Ah! The horror! 🙀

Well, you'll be happy to know that there is a simpler solution to this, "*Custom Environments in Gym*". Yes, you read that right! Gym allows you to add custom environments to enable you to call it with gym.make and use all the standard functions that we discussed in the [previous post](https://www.saashanair.com/introduction-to-openai-gym/).

### Associated Video

{{< youtube kd4RrN-FTWY >}}

### Steps for adding a custom environment:

For this post, instead of creating our own environment, let's use the CartPole environment available from Gym (code available [here](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)), which we save locally as `custom_cartpole_env.py`. To help us identify that we are actually using the custom cart pole environment, and not the standard one provided with the Gym installation, rename the class in custom_cartpole_env.py to `CustomCartPoleEnv`. Additionally, in the render function defined within the CustomCartPoleEnv class, modify the colours of the components. I went with blue for the cart, green for the pole, black for axle and red for the track. 🤪

Before we get into the procedure for adding the customised cartpole environment to Gym, take a moment to scroll through the structure of the custom_cartpole_env.py file. You would notice that the CustomCartPoleEnv class inherits from `gym.Env` , and contains functions for `reset(self)` , `step(self, action)` , `render(self)`  and `close(self)` and variables for `action_space` and `observation_space` ( [types of Spaces available in Gym](https://github.com/openai/gym/tree/master/gym/spaces)). Thus, when you wish to add a custom environment, say TORCS, to Gym, you would need to create a wrapper with a similar structure which abstracts over the specifics of interacting with the environment.

***Step 1: Replicate the folder structure*** shown below. I went with the name `custom_envs` for the parent package, where we can create folders for multiple environments. Within the parent package, we have `custom_cartpole`, which in turns contains the `envs` folder. Feel free to rename the folders as per your needs and/or whims.

```
custom_envs/
  |_ setup.py
  |_ custom_cartpole/
      |_ __init__.py
      |_ envs/
          |_ __init__.py
          |_ custom_cartpole_env.py
```

***Step 2: Edit the [setup.py](http://setup.py/) file***

```python
from setuptools import setup

setup(name='custom_envs', # name of the package
	version='0.0.1', # version of this release
	install_requires=['gym'] # specifies the minimal list of libraries required to run the package correctly
)
```

The setup file makes it convenient to build and distribute your packages. The first two arguments listed above, '*name*' and '*version*' are required arguments of the `setup()` function. The third argument, '*install_requires*' is most important for us. It specifies the complete  list of libraries that need to be installed to ensure proper usage of the newly created package, thus saving hours of manual labour in setup (read more about it [here](https://packaging.python.org/discussions/install-requires-vs-requirements/)). For the custom cartpole environment, we only need Gym, but, based on your needs you could add more items to the list.

***Step 3: Edit the custom_cartpole/\*\*init\*\*.py file***

```python
from gym.envs.registration import register

register(id='CustomCartPole-v0', # id by which to refer to the new environment; the string is passed as an argument to gym.make() to create a copy of the environment
	entry_point='custom_cartpole.envs:CustomCartPoleEnv' # points to the class that inherits from gym.Env and defines the four basic functions, i.e. reset, step, render, close
)
```

Gym tracks the available environments by maintaining a Registry which maps a unique ID to the associated specifications. To be able to call your environment using the gym.make function, it is essential to register your custom environment with the local copy of the registry present on your system. To do so, Gym provides the register function (you can dig deeper into the specifics [here](https://github.com/openai/gym/blob/master/gym/envs/registration.py)), which accepts two arguments, namely, `id` and `entry_point`. The id specifies the string to be passed to the gym.make function to access the particular environment, it has to be of the form '*name*' followed by '*-vX*' as indicated in the example above. Entry point specifies the location of the class that inherits from gym.Env and contains definitions for the basic functions, i.e. reset, step, render, close (here, the CustomCartPoleEnv class).

***Step 4: Edit the custom_cartpole/envs/\*\*init\*\*.py file***

```python
from custom_cartpole.envs.custom_cartpole_env import CustomCartPoleEnv # points to the location where the class that inherits from gym.Env can be found
```

Since the previous init file directly calls upon the CustomCartPoleEnv without bothering to locate it within the directory structure, this init file is used to indicate information regarding the name of the file within which the entry point class, here CustomCartPoleEnv, is located.

***Step 5: Installing the newly created package***

In the terminal, navigate to the parent folder where the [setup.py](http://setup.py/) file is located, and type the following:

```bash
pip install -e .
```

This command uses the specifications defined within the [setup.py](http://setup.py/) file and prepares your workspace to enable working with the newly added custom environment.

***Step 6: Testing your custom environment***

Everything is ready to go, and now it is just a matter of modifying the skeleton code from the [previous post](https://www.saashanair.com/introduction-to-openai-gym/) to call `CustomCartPole-v0` instead of `CartPole-v0`.

```python
import gym
from gym.wrappers import Monitor

#env = gym.make('CartPole-v0')

import custom_cartpole
env = gym.make('CustomCartPole-v0')

max_ep = 10

for ep_cnt in range(max_ep):
	step_cnt = 0
	ep_reward = 0
	done = False
	state = env.reset()

	while not done:
		next_state, reward, done, _ = env.step(env.action_space.sample())
		env.render()
		step_cnt += 1
		ep_reward += reward
		state = next_state

	print('Episode: {}, Step count: {}, Episode reward: {}'.format(ep_cnt, step_cnt, ep_reward))
env.close()
```

Congratulations, you have successfully added a custom environment to the gym registry! 🎉 The colours of the custom cartpole environment look horrendous, but at least we know that the whole shebang works. 🤪


<img src="/images/posts/20200926_openai_gym_and_custom_env/1_random_agent.gif" class="large" alt="An episode of interactions of a random agent with the Custom CartPole environment">
<em>An episode of interactions of a random agent with the Custom CartPole environment</em>


An episode of interactions of a random agent with the Custom CartPole environment

Dear reader, hope you found this post useful. I am looking forward to seeing how you use this setup for your projects and the custom environments that you build, so please do drop me a message at [saasha.allthingsai@gmail.com](mailto:saasha.allthingsai@gmail.com), or hit me up on [Twitter](https://twitter.com/saasha_nair). 💙

See you. 👋

---

### Resources

1. [Example from OpenAI Gym for adding a multi-agent Soccer environment to the registry](https://github.com/openai/gym-soccer)
2. [Official Documentation explaining how to add a new environment to OpenAI Gym](https://github.com/openai/gym/blob/master/docs/creating-environments.md)
3. [Script containing details of the register function](https://github.com/openai/gym/blob/master/gym/envs/registration.py)
4. [Different types of Spaces available in Gym for use with action_space and obsevation_space](https://github.com/openai/gym/tree/master/gym/spaces)
5. [Install_requires in setup() vs Requirements file](https://packaging.python.org/discussions/install-requires-vs-requirements/)
6. [Code for the OpenAI Gym's CartPole environment](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)