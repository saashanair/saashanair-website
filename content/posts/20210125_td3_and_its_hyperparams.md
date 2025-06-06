---
title: "TD3 and its Hyperparameters"
date: 2021-01-25
tags: "Reinforcement Learning"
math: true
---

**NOTE 1**: This is the final post in a three-part series on the Twin Delayed DDPG (TD3) algorithm.

- {{< backlink "20210112_td3_theory" "Part 1">}}: theory explaining the different components that build up the algorithm.
- {{< backlink "20210118_td3_algo_to_code" "Part 2">}}: how the algorithm is translated to code.
- Part 3: how different hyperparameters affect the behaviour of the algorithm.

**NOTE 2:** The complete code discussed in this series of posts can be found [here](https://github.com/saashanair/rl-series/tree/master/td3) on Github.

---

{{< toc >}}

---

The success of TD3 can be attributed to three main factors, namely, 'Clipped Double Q-learning', 'Delayed Policy and Target Updates' and 'Target Policy Smoothing' (see {{< backlink "20210112_td3_theory" "first post of this series">}} for more details). How important are these three factors for an agent to learn? Let's find out in this post!

Hi dear reader! Hope you have enjoyed the journey thus far. 🤗

In this final segment of the series on mastering TD3, we will work our way through the main hyperparameters that drive the algorithm to observe how the behaviour of the algorithm changes when the values deviate from those suggested in the [paper](https://arxiv.org/pdf/1802.09477.pdf). All results discussed below are based on training the {{< backlink "20210118_td3_algo_to_code" "two-headed critic implementation (from the previous post)">}}. The agents were trained for 1000 episodes on the LunarLanderContinuous-v2 environment with actor and critic learning rates of 1e-4. The training episodes were reduced from 3000 in the previous post to 1000 in this one to reduce training time (even with just 1000 episodes it took about 2 hours to train on a GPU!). Now that we have the settings for the experiments all straightened out, let's dive right in!

## Experiment 1: Effect of Target Policy Smoothing ($\widetilde{\sigma}$ and $c$)

In the paper, the authors note that 'Target Policy Smoothing' is added to reduce the variance of the learned policies, to make them less brittle. The paper suggests using a small Gaussian noise with a standard deviation of 0.2 being clipped within the range [-0.5, +0.5]. But how is the policy affected when we move towards either of the extremes, i.e. 'no smoothing' vs 'too much smoothing'?

<img src="/images/posts/20210125_td3_and_its_hyperparams/varying-smoothing.png" class="large" alt="">
<em>Fig 1: Effect of varying the magnitude of Gaussian noise applied for 'Target Policy Smoothing'</em>


As one would guess, the agent still learns when no noise is added to the target policy. However, the policy suffers from high variance. The peaks of the shaded band around the mean trend, not only jump as high as 1000, but also drop below the 200 point threshold. Conversely, adding too much noise  to the target policy, by setting the standard deviation to 1.0 and clipping value within the range [-1.0, + 1.0], damages the learning of the agent, rendering it completely useless.

## Experiment 2: Varying the update frequency ($d$)

The paper suggests an update frequency of 2, which means that the two critic-networks are updated at each learning step, while the actor-network and the three target-networks are updated only on alternate learning steps. What do you reckon would happen if we set $d$ such that the actor and targets update with each step, or if we move in the opposite direction by increasing the $d$ value to decrease the frequency of updates?

<img src="/images/posts/20210125_td3_and_its_hyperparams/varying-d.png" class="large" alt="">
<em>Fig 2: Effect of varying the update frequency ($d$) used for 'Delayed Policy and Target Updates'</em>


Again, it is helpful to note here that, similar to 'Target Policy Smoothing', 'Delayed Updates' was added to reduce variance in the learned policy. The graph above helps you see that $d=2$ (default) gives the most stable policy (i.e., the test-time graph does not exhibit sudden rises and falls). On the other hand, setting $d$ to values between 1 to 10 (other than 2) does not completely wreck the learning curve, it does result in wildly fluctuating test-time performance (indicated by the spread of the shaded region).

## Experiment 3: Is the delayed update important for both, the actor and the targets?

As per the paper, we need to implement delayed updates to not just the target-networks (as is done in DQN), but also to the actor-network. Could we get away with not applying the delay to the actor, but only to the three target-networks?

<img src="/images/posts/20210125_td3_and_its_hyperparams/delayed-updates.png" class="large" alt="">
<em>Fig 3: Effect of applying delayed updates only to the target-networks, compared to the default setting applying delayed updates also to the actor.</em>


Again, the agent with delayed updates applied only to the targets behaves as expected, it still learns to solve the task at hand, but with extremely noisy test-time behaviour. This makes complete sense. As discussed in the {{< backlink "20210112_td3_theory" "first post in this series">}}, the reason to delay the update of the actor was to allow the Q-value estimates of the critics to converge, thereby reducing variance in the learned policy, and so the noisy behaviour is expected.

## Experiment 4: Effect of Clipped Double Q-learning (CDQ)

Based on the experiments above, my hypothesis is that clipped double q-learning is the essence of the TD3 algorithm. The other two factors, though important for stability, do not have much of an impact on the performance of the algorithm. To test this out, let's compare the performance of the algorithm when trained with the default values against a model trained with only 'Clipped Double Q-learning' (i.e, 'Target Policy Smoothing' and 'Delay Updates of Actor and Targets' are not applied).

<img src="/images/posts/20210125_td3_and_its_hyperparams/effect-cdq.png" class="large" alt="">
<em>Fig 4: Effect of applying only Clipped Double Q-learning (CDQ), while keeping the Delayed Updates and Target Policy Smoothing turned off.</em>


Though the agent with only 'Clipped Double Q-learning' appears to cross the rewards threshold during training, it underperforms during testing. A potential cause for this is that the model is overfitting certain values within the vast continuous action space, and that is where 'Target Policy Smoothing' fits in.

## Experiment 5: Varying the tau values in target update

The paper stresses the need for slow updates of the target-networks using Polyak averaging with a $\tau$ value of 0.005. What then is the effect of performing a hard update?

<img src="/images/posts/20210125_td3_and_its_hyperparams/varying-tau.png" class="large" alt="">
<em>Fig 5: Effect of varying the $\tau$ value used for slow update of target-networks</em>


Setting $\tau=1$ causes a hard update of the weights (i.e., the weights of the actor and critics are directly copied over to the corresponding target-networks). This has a drastically negative effect on the agent's learning. In fact, the learning already begins to deteriorate when $\tau$ is set to 0.1, which is much larger than the value of 0.005 suggested in the paper. The thing that surprised me though was setting $\tau$ to 0.001 (the value suggested in the DDPG paper) did not affect the agent's learning, but caused the learned policy to be extremely noisy.  Dear reader, if you do have any insights on this, please do share them with me. 🙃

## Experiment 6: Varying the amount of stochasticity during training

As we had {{< backlink "20210112_td3_theory" "fdiscussed earlier">}}, though TD3 learns a deterministic policy, it explores the environment using a stochastic behaviour policy. This is achieved by adding a small Gaussian noise to the actions performed by the agent during training. So what should the magnitude of the noise be for the agent to be able to learn well?

<img src="/images/posts/20210125_td3_and_its_hyperparams/varying-expl.png" class="large" alt="">
<em>Fig 6: Effect of varying the magnitude of Gaussian noise applied for encouraging exploration during training</em>


The paper suggests a small zero-mean Gaussian noise with a standard deviation of 0.1, and as one would imagine, on both sides of that value the performance drops. The training-time plots of the exploration noise other than 0.1 are unable to cross the reward threshold. During test-time, the corresponding agents show extremely noisy behaviour with a wildly fluctuating mean trend (the dark line) and a thick shaded region dropping far below the reward threshold. Conversely, the agent with an exploration noise of 0.1, as suggested in the paper, has a tight shaded region that appears to be completely above the (red) threshold line. A potential cause for this behaviour can be attributed to the agents learning the noise when the amount of perturbation is too high, and overfitting to suboptimal actions when the magnitude of applied noise is too low.

## Ciao

![](https://media.giphy.com/media/lTpme2Po0hkqI/giphy.gif)

All right then folks, that's it for today. Hope you found this series on TD3 helpful. 😄

Do let me know what you thought of the new format of multiple posts on the same algorithm, instead of a single really long post. I am always available to chat via email at [saasha.allthingsai@gmail.com](mailto:saasha.allthingsai@gmail.com) or on [Twitter](https://twitter.com/saasha_nair). 💌

See you soon and take care.

---

## Further reading

1. [TD3 Implementation](https://github.com/sfujim/TD3), by Scott Fujimoto -- code by the author of the paper on TD3
2. [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf), Scott Fujimoto et. al. -- the paper on TD3