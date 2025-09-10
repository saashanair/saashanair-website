---
title: "AI Agents: What is All the Hype About?"
date: 2025-09-10
tags: [sufficiently explained, ai]
---

You cannot seem to go online today without someone telling you how *Agentic AI* is going to revolutionise the world. AI agents, the claim goes, are the next big frontier in AI. But what even does that mean?

## What Do We Mean by an “Agent”?

The term may feel new, but it has been around for decades. In [*Artificial Intelligence: A Modern Approach*, Stuart Russell and Peter Norvig](https://people.engr.tamu.edu/guni/csce642/files/AI_Russell_Norvig.pdf) define an agent as *“anything that can be viewed as perceiving its environment through sensors and acting upon that environment through effectors.”*

This definition is easiest to picture in robotics. Sensors such as cameras, infrared detectors, or LiDAR allow the robot to perceive the world around it, while robotic arms or grippers act as effectors, enabling it to perform physical actions.

More generally, we can think of an agent as a system that perceives its environment, plans, and then acts by executing the set of actions available to it. Here:

- **Environments** are the domains where agents operate. For a Mario-playing agent, the environment is the video game. For a Roomba, it is your living room.
- **Set of actions** defines what the agent can do within its environment. Mario can move left, right, or jump. A Roomba can roll across the floor and vacuum dust.

Agents can operate alone or in groups. A single agent might vacuum your house. A set of agents might coordinate, such as fleets of autonomous vehicles communicating to avoid collisions and optimise routes.

So if the concept is decades old, why are agents suddenly the buzzword of 2025?

## Why Now? Enter LLMs

The recent success of **large language models (LLMs)** has given agents new life. LLMs make it possible for agents to:

- **Be instructed in plain English**, making them easy to set up.
- **Be deployed across multiple environments** without the need to handcraft rules for each.
- **Chain reasoning and tool use** in ways that feel natural to humans.

This shift means we can now build agents that are general-purpose, conversational, and usable by both technical and non-technical people. Even though [recent research suggests smaller specialised models may perform better as agents](https://arxiv.org/pdf/2506.02153), LLMs have been the accelerant for today’s “agentic” moment.

In this LLM-powered setup, the agent’s environment is often digital rather than physical: your email and calendar system, or enterprise tools such as Jira and Confluence. Essentially, the environment is where the agent gathers information and decides its next move. Agents then act through tools, interacting with APIs, databases, or applications that let them perform meaningful work, such as retrieving data, sending emails, or scheduling meetings.

{{< newsletter_signup_blogpost >}}

## Making Sense of Today’s Agents: Autonomy as a Spectrum

The problem is that the word “agent” is overloaded. Depending on who you ask, it could mean a chatbot, a workflow automation system, or a fully autonomous digital worker. To cut through the noise, I find it useful to think about agents in terms of **levels of autonomy**.

<img src="/images/posts/20250910_ai_agents_what_is_all_the_hype_about/ai_agent_autonomy.png" class="large" alt="">
<em>AI agents exist on a spectrum of autonomy</em>

⚠️ Caveat: this is not a standard taxonomy. The field is still evolving. It is simply a mental model I find useful to keep things straight in my own head.


### **Workflow Orchestrators (high autonomy)**

On one end, you have the *visionary* AI Agents (or Agentic AI): systems that can run entire workflows end to end. Workflow orchestrators can chain multiple tools and actions to execute complex sequences, effectively acting in the world on your behalf.

Example: you say, *“Launch a new e-commerce store for me.”* The agent registers the domain, designs the website, uploads products, and sets up marketing campaigns, all without you lifting a finger.

This is the holy grail: agents as ‘virtual coworkers’ you can delegate tasks to. Though we are not quite there yet for consumer use cases, we are already seeing glimmers of promise in research:

- [**The AI Scientist**](https://sakana.ai/ai-scientist/) by Sakana AI and researchers from the Universities of Oxford and British Columbia can fully automate scientific discovery. It generates novel research ideas, designs and executes experiments, visualises results, and writes full scientific papers.
- [**AlphaEvolve**](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) by Google DeepMind autonomously improves algorithms by iterating through ideation, experiment design, testing, and evaluation, demonstrating potential for practical innovations and scientific discoveries.

These are early but striking examples of what highly autonomous workflow orchestrators could look like in practice: not just assistants, but genuine collaborators.


### **Task Assistants (medium autonomy)**

In the middle are *task-specific agents*. They can execute well-defined actions but usually require check-ins or confirmation.

Example: An Expense Management assistant that is tasked with categorising receipts and submitting them for reimbursement. The agent scans documents provided by you, allocates them to the correct expense categories, and asks for confirmation before submission.

These assistants are powerful, but narrow. They help with discrete tasks rather than whole processes. Think of them like adaptive cruise control in cars: helpful, but you are still in charge.

### **Knowledge Assistants (low autonomy)**

At the other end are the simplest (and currently most common) agents: chatbots connected to a restricted knowledge base.

Example: an HR assistant that answers questions like: *“What is the process for applying for leave?”* or *“How do I submit an expense request?”*

These agents do not take actions in the world, but they add real value by saving employees time, reducing friction, and being available 24/7.

You can even set this up today from your browser, no technical skills needed. Tools such as Microsoft Copilot (see [tutorial](https://www.youtube.com/watch?v=211EGT_2x9c)) and Gemini Gems (see [tutorial](https://www.youtube.com/watch?v=yO01B8OoXfo)) make it straightforward to spin up your own knowledge assistant.

## The State of Play

While the vision of highly autonomous workflow orchestrators is exciting, the reality today is more modest. Most agents in the wild are still *knowledge assistants* and *task-specific helpers*.

Enterprises are experimenting, but cautiously. [**McKinsey** notes that despite the surge of interest in “agentic AI”, most deployments remain small-scale pilots](https://www.mckinsey.com/~/media/mckinsey/business%20functions/mckinsey%20digital/our%20insights/the%20top%20trends%20in%20tech%202025/mckinsey-technology-trends-outlook-2025.pdf?utm_source=chatgpt.com) focused on *specific, high-value business problems*, rather than scaled rollouts.

[**Gartner** echoes this view, pointing to a wide gap](https://www.gartner.com/en/articles/intelligent-agent-in-ai?utm_source=chatgpt.com) between today’s LLM-powered assistants and the aspirational vision of fully autonomous agents. They estimate that [**over 40% of agentic AI projects may be cancelled by 2027**](https://www.reuters.com/business/over-40-agentic-ai-projects-will-be-scrapped-by-2027-gartner-says-2025-06-25/?utm_source=chatgpt.com) due to high costs and unclear business value. Yet this experimentation is not wasted. By 2028, Gartner projects around **one-third of enterprise applications will include agentic AI**, up from less than 1% in 2024.


## The Bottom Line

Agents are not new. What is new is that LLMs have made them accessible, versatile, and suddenly relevant to everyday workflows. But autonomy still comes in levels: from knowledge assistants that simply answer questions, to workflow orchestrators that hint at the future of digital coworkers. Despite current limitations, the groundwork being laid today hints at a future where AI agents are true collaborators. The experiments underway are laying the foundation for the next leap.
