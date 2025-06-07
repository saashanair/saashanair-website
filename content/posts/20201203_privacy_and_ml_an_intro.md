---
title: "Privacy and ML: An Introduction"
date: 2020-12-03
tags: "Misc Tech"
math: true
---


Generating, storing and analysing massive amounts of data has not only become easy but also inexpensive. This has prompted organisations to collect and store information about every user interaction, ranging from a click to a scroll to read time and so on, with the product/service, in the hope that even if not of much use today, it would help generate some insights in the future. In some ways, the digital economy has turned tech companies and data brokers into *'data hoarders'* and us, the users, into their live *'data points'*.

## Private data is a toxic asset

In today's economy, organisations use our private data to segment us, while data brokers sell our private data to profit off us, which seems unfair. This has led [Bruce Schneier](https://www.schneier.com/blog/about/) to deem [data as a toxic asset](https://www.schneier.com/blog/archives/2016/03/data_is_a_toxic.html). Or as [Dr. Carissa Véliz](https://www.carissaveliz.com/) describes in her book [*'Privacy is Power: Why and how you should take back control of your data'*](https://www.carissaveliz.com/books)(my notes {{< backlink "20210103_privacy_is_power" "here">}}),

> Personal data is dangerous because it is sensitive, highly susceptible to misuse, hard to keep safe and desired by many.

The misuse of private information is not a new phenomenon. During World War II, [census data on the population was used to identify minority groups](https://medium.com/@hansdezwart/during-world-war-ii-we-did-have-something-to-hide-40689565c550). Ofcourse, this is an extreme example of data misuse, but as Dr. Véliz and many others in the field argue, if we continue down the current trajectory, we might soon find ourselves in a state of 'Digital Totalitarianism'.

## Examples of privacy attacks in Deep Learning research

If you are from an ML background, there might be a part of you that is still not willing to accept that data can do bad. After all, data forms an integral part of our livelihoods. You might think, what if we created the ML models and then encrypted (or dare I say, deleted) the data right away? Should that not solve the problem? It turns out, our favourite ML and DL techniques can quite as easily be hacked. Focusing particularly on the current Deep Learning trend, Neural Networks are often overparameterized and have a tendency to memorise idiosyncrasies existing in the training data, opening vulnerabilities for exploitation.

<img src="/images/posts/20201203_privacy_and_ml_an_intro/fig1.png" class="large" alt="">
<em>Fig. 1: Example of a model inversion attack on a Face Recognition API. Left: Resconstructed image; Right: Original image; (<a href="https://www.cs.cmu.edu/~mfredrik/papers/fjr2015ccs.pdf">source</a>)</em>

Not yet convinced? Let's look at this work by Fredrikson et. al. titled [*'Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures'*](https://www.cs.cmu.edu/~mfredrik/papers/fjr2015ccs.pdf). Model Inversion attacks are those that attempt to reconstruct the feature vectors that are used to create a particular model (for a list of the different types of attack, read [this](https://arxiv.org/pdf/2004.12254.pdf)). The authors of the paper show that given black box access to a model that returns confidence scores at the output, one can easily reverse-engineer the input by posing it as an optimisation problem with the objective of maximising the confidence score of the target class to 1.0. To show the impact of this, the authors use a Face Recognition API and are able to reconstruct an average representation of the faces of their "victims" knowing only their names. As you can see in Fig. 1 above, the reconstructed image is quite blurry, which the authors recognise in the paper as well. Thus, to prove the efficacy of their technique, the authors use Amazon's Mechanical Turk to test if humans can pick the target person in a lineup using only the recovered image, and they achieved a performance of 80% across all workers, with accuracy going up to even 95% for some workers. Agreed that we skipped over the specifics of the technique and the nuances involved, but this still gave me the same kind of feeling that I had when I watched season 1 of "You" on Netflix!

<img src="/images/posts/20201203_privacy_and_ml_an_intro/fig2.png" class="large" alt="">
<em>Fig. 2: De-anonymization of Netflix dataset using auxiliary IMDb data (<a href="http://www.gautamkamath.com/CS860notes/lec1.pdf">source</a>)</em>

The cynic in you might say, of course they were able to perform the attack, they knew the name of the person. So let's look at another example. A common technique quoted for protecting privacy is to remove all identifying features from the dataset such as name, age, address and so on. Even Netflix thought so when they released the dataset for the Netflix Prize back in 2006 saying “No, all customer identifying information has been removed; all that remains are ratings and dates. This follows our privacy policy. Even if, for example, you knew all your own ratings and their dates you probably couldn’t identify them reliably in the data because only a small sample was included (less than one- tenth of our complete dataset) and that data was subject to perturbation. Of course, since you know all your own ratings that really isn’t a privacy problem is it?”([source](https://www.netflixprize.com/faq.html)). But in the widely cited paper by Narayanan and Shmatikov titled [*'Robust De-anonymization of Large Sparse Datasets'*](https://www.cs.utexas.edu/~shmat/shmat_oak08netflix.pdf), as illustrated in Fig. 2, the authors were able to use public IMDb data posted voluntarily by users to trace the identities of the individuals who formed a part of the released Netflix dataset. Given that all of us are voluntarily posting data about ourselves online, it isn't very difficult to find such auxiliary data even for other use cases.

## Setting up a running example

Security and privacy research is full of many examples of such attacks. But you might say you don't need to worry about any of this as your work does not involve the use of such data. Considering that I work in the space of autonomous/highly automated vehicles, let's setup a running example with that as the usecase.

<img src="/images/posts/20201203_privacy_and_ml_an_intro/fig3.jpeg" class="large" alt="">
<em>Fig. 3: Data collected by the ADAS module used as a running example for this post</em>

Imagine an advanced driving assistance system (ADAS) deployed on a car with multiple sensors. Such systems usually collect driving data to ensure that their assistance capabilities can be improved over time. Consider that your ADAS, among other data, is collecting the following, *<current location, current time, sensor readings at current time, action performed at current time>*. Similar data is being collected across tens of thousands of vehicles.

<img src="/images/posts/20201203_privacy_and_ml_an_intro/fig4.gif" class="large" alt="">
<em>Fig. 4: Illustrating a naive approach to learning on data that does not preserve user privacy</em>

The most naive way of training on this data (as illustrated in Fig. 4 above) would be to pass the data from all the vehicles to a central server, which performs computations to train/update the ADAS model. This trained model can either be passed back to the vehicles or be available to the vehicles to query via API calls.

<img src="/images/posts/20201203_privacy_and_ml_an_intro/fig5.jpeg" class="large" alt="">
<em>Fig. 5: Potential attack surface for the naive approach which moves user data to a central server</em>


As you could imagine, this setup leaves a lot of possible attack surfaces that can be exploited. With all the data being saved in one central location, an adversary could just as easily gain access to it. The *"current location"* information could be used to infer the home addresses and travel schedules of vehicle owners to know when they would not be at home for targeted robbery. Or the data could be sold by the car manufactures to insurance companies, who could in turn infer your driving behaviour, and jack up your premiums if you are an obnoxious driver. If you can think of any other such concrete examples, I'd love to hear them in the comments below.

## Is there anything we can do?

After the grimness of the above sections, I would not blame you if you say I am not giving my data to anyone. What is the point, it is just going to be misused or exploited anyway. I like the outlook [Dr. Helen Nissenbaum](https://nissenbaum.tech.cornell.edu/) presented on this at *PriCon 2020* (read a [summary of the talk](https://blog.openmined.org/conference-talk-summary-helen-nissenbaum-privacy-contextual-integrity-and-obfuscation/)), where she said,

> Privacy is about the appropriate flow of information, not about locking up data in vaults.

A doctor-patient relation helps put this point in perspective. While the doctor is not to share any personal information (read data) about the patient with anyone, he/she may use the knowledge gained from such data to treat other patients with similar symptoms. This responsible use of personal data by the doctor leads to the betterment of society without compromising the privacy of any individual.

In the next two sections, we will look at two techniques for preserving privacy in ML that have been gaining traction lately.

## Technique 1: Federated Learning

Going back to the naive solution for training the ADAS model from earlier, the problem was that we were moving the data to the server, providing a central area of attack. What if instead of bringing the data to the models, we could bring the models to the data? This forms the premise of Federated Learning.

More formally, Federated Learning can be defined as "a machine learning setting where **many clients** (eg. mobile devices or whole organisations) **collaboratively train** a model under the **orchestration of a central server**, while keeping the training **data decentralised**"([source](https://arxiv.org/pdf/1912.04977.pdf)).

As with any tool, you can't just throw Federated Learning at the problem and expect it to work. Federated Learning would be the right choice if the [following criteria](https://arxiv.org/pdf/1602.05629.pdf) are met:

- training on real-world data on edge devices provides a distinct advantage over training on proxy data available at the data centre.
- data is privacy sensitive and/or large in size.
- user interaction allows the labels of the data to be naturally inferred.

In the case of the example with the ADAS model established earlier all three points are met. We would have tens of thousands of vehicles to learn from, and data generated on the road is better and more realistic than data generated in a simulator. Additionally, as discussed earlier, we need to maintain the privacy of the data collected on each vehicle. As for inferring labels, the action performed would act as a label for each of the sensor readings indicating the scenario that the vehicle is currently in. Thus, federated learning would be a good match for our use case.

So, what then does the [setting for a typical FL algorithm](https://arxiv.org/pdf/1602.05629.pdf) look like? The characteristics can be listed as:

- **Non-IID data:** Each local dataset is specific to a particular user and does not represent the entire population. For the ADAS model, a particular individual dataset might show that the person is an obnoxious driver, but that does not represent the driving behaviour across the entire population of drivers that would be part of the test set.
- **Unbalanced:** The number of samples across local datasets could vary depending on the amount of use. The amount of data collected by the ADAS model on a person who drives to work everyday would be much more when compared to an infrequent user.
- **Massively distributed:** The number of clients outnumber samples per client. With tens of thousands of cars on the road, this is bound to be true.
- **Limited communication:** This can result from the devices being turned off or due to unreliable/expensive connections. For example, if the model is set to train at 1700hrs everyday, the weekend driver's vehicle would usually be unavailable for training from Monday-Friday.

As one can guess from looking at the points listed above, there exist many complications such as, how to design the network architecture without ever looking at the data, how to most efficiently train models on edge devices while limiting computation costs, how to efficiently transfer models between the server and the edge devices and so on. Being a nascent field, it is constantly growing and evolving, and thus provides scope for contributions (for a list of open problems in the field, read [this paper](https://arxiv.org/pdf/1912.04977.pdf)).

### FedAvg

Moving to a more concrete example of the algorithm, [FederatedAveraging](https://arxiv.org/pdf/1602.05629.pdf) was one of the initial FL algorithms introduced back in 2016, and forms the basis of many FL algorithms used today. Fig. 6 below helps build an intuition for the working of the FedAvg algorithm.

<img src="/images/posts/20201203_privacy_and_ml_an_intro/fig6.mp4" class="large" alt="">
<em>Fig 6: An intuitive illustration of the FedAvg algorithm</em>

FedAvg works by assuming a fixed set of clients `K` that work towards updating the global model in `t` training rounds of synchronous communication. In each round `t`, the server picks a fraction `C` of clients to train on and a copy of the global model is then sent to the selected clients. Each of the clients then use their local dataset to compute gradients with respect to the global model, these gradients are then sent back to the server. To reduce the amount of communication between the server and clients over expensive/unreliable mediums (see the previous section), each of the clients accumulate gradients over `E` local epochs before sending the gradients back to the server. The server then updates the global model by applying a weighted average of each of the received gradients. The weighted average helps accommodate for the variations in the amount of samples across each of the clients.

<img src="/images/posts/20201203_privacy_and_ml_an_intro/fig7.jpeg" class="large" alt="">
<em>Fig. 7: Potential attack surfaces for the FedAvg algorithm</em>


On further inspecting the new setup, we would notice that though we are definitely in a better position compared to the initial naive approach, there still exist many potential attack surfaces, as shown in Fig. 7 above. The adversary still has access to the model which could be reverse-engineered. Moreover, the gradient updates could be leaked. Though the gradient is better than transporting raw data between the client and server, it is still a *"compressed form"* of the data, and could contain private information. Also, if the raw data is stored at the client for a long period, it could be a potential attack point.

## Technique 2: Differential Privacy

As we saw in the previous section, having raw data or even just its gradients can pose a potential risk. Differential Privacy is a framework that suggests adding a small amount of noise to the process, either in the data, model or outputs depending on the chosen algorithm, with the aim of obfuscating private data.

<img src="/images/posts/20201203_privacy_and_ml_an_intro/fig8.gif" class="large" alt="">
<em>Fig. 8: An illustration indicating the definition of Differential Privacy</em>


As per [Differential Privacy: A Primer for a Non-technical Audience](https://privacytools.seas.harvard.edu/files/privacytools/files/pedagogical-document-dp_0.pdf), *"DP provides mathematical guarantees that anyone seeing the result of a differentially private analysis will essentially make the same inference about any individual’s private information, whether or not that individual’s private information is included in the input to the analysis"*. This means, as shown in Fig. 8 above, the output of performing inference on a certain model should be almost similar when the entire training data is used or if samples of a single individual are excluded from training. In simpler terms, DP helps ensure that the model learns the patterns from the data and not the idiosyncrasies of individuals who exist in the dataset. This can be mathematically represented as shown in Fig. 9 below.

<img src="/images/posts/20201203_privacy_and_ml_an_intro/fig9.jpeg" class="large" alt="">
<em>Fig. 9: Mathematical representation of the definition of Differential Privacy (<a href="https://arxiv.org/pdf/1412.7584.pdf">source</a>)</em>


Though this post won't delve deep into the mathematics of DP (if interested, do take a look at this wonderfully detailed [series of posts](https://desfontain.es/privacy/differential-privacy-awesomeness.html)), an important property to note is that of "composition". Suppose two queries are performed on the same dataset, such that the privacy loss applied per query amounts to $\epsilon$, then the total privacy loss parameter on the dataset now accounts to $2*\epsilon$. Thus, with multiple queries to the dataset, though differential privacy is still preserved, the $\epsilon$ values continue to add up and the privacy of the dataset continues to get weaker. 

### Private Aggregation of Teacher Ensembles (PATE)

For a more concrete understanding of how to apply DP to Deep Learning, let's take a look at [PATE](https://arxiv.org/pdf/1610.05755.pdf).

<img src="/images/posts/20201203_privacy_and_ml_an_intro/fig10.mp4" class="large" alt="">
<em>Fig. 10: An intuitive illustration of the PATE algorithm (Based on the illustrations in '<a href="http://www.cleverhans.io/privacy/2018/04/29/privacy-and-machine-learning.html">Privacy and Machine Learning: Two Unexpected Allies?</a>' by Papernot and Goodfellow)</em>


The algorithm works by first creating disjointed partitions of the entire dataset, such that all data about a particular individual belongs to the same partition. In the Fig. 10 above, we create three partitions, each one relating to one of our example vehicles. Ofcourse, in a real-world setting the partitions would be much larger, and each partition would contain data about multiple vehicles. Each of the disjointed partitions has a neural network model associated with it, which are collectively referred to as the "teachers". Each of the teacher models are trained only on the data contained within their partitions. This ensemble of trained teachers can then be used to make predictions. Thus, at inference, the new sample to be labeled is passed through each of the models, and the generated predictions are gathered. In a non-DP setting, we would return the label with the highest count as the response of the model. However, to ensure that privacy is preserved, an additional step of applying noise to the teacher votes is carried out, following which the label with the highest "noisy counts" is returned as the prediction. You might wonder, what is the significance of this setup with DP? Well, if you took a step back and re-examined the algorithm, you would notice that there are two possible outcomes at this stage. One is that, in the teacher vote count stage, majority of the models are in consensus and predict the same label, since each of the teachers were trained on different subsets of the data, it implies that the teachers were able to extract general patterns in the dataset. Thus, only a small amount of noise controlled by the $\epsilon$ value needs to be applied to the output. The other possible outcome is when no clear majority exists, i.e. a situation with a high probability of a potential data leak. Here, the algorithm would need to expend a higher $\epsilon$ value to add a bigger noise to ensure that privacy is preserved.

Though the algorithm could stop at this stage and the deployed ensembles would be privacy preserving, the authors of the paper perform one additional step (illustrated in Fig. 10 as Step 3). They assume the availability of a public unlabelled dataset, to which the ensemble is applied to generate labels with privacy. The newly labeled dataset is then used in a supervised setting to train a new model, called the "student". At the time of deployment, only the student model is used, and the teacher models are no longer accessible. This extra step in fact turns out to be extremely important and advantageous. As previously discussed, the $\epsilon$ value adds up with each query performed on the dataset. This means that either the number of inferences that the ensemble is allowed to perform would have to be restricted or else, we risk information leak as $\epsilon$ gets larger. Additionally, if the ensemble is accessible, an adversary could perform a model inversion attack (discussed earlier) to extract private data captured by each of the teachers. Both of these problems are easily avoided by making the teachers inaccessible and deploying only the student.

## Conclusion

This post attempts to convince you of the importance of privacy by showing how private data can be used for discrimination, manipulation, identity theft and so on. Thus, as ML researchers and practitioners, we need to be mindful of the consequences of the data we collect and the models we create and deploy. This post barely scratches the surface of the current landscape of privacy technology in ML. This was only a brief introduction to Federated Learning and Differential Privacy.There are multiple other potential topics to dive into, such as Homomorphic Encryption, Secure Multiparty Computation, or some combination of these, such as Federated Learning with Differential Privacy. Let me know in the comments below, what you work on and which of these topics are of interest to you.

Thank you, dear reader, for sticking through this long post, hope you found it helpful. Looking forward to hearing from you. Drop me a message at [saasha.allthingsai@gmail.com](mailto:saasha.allthingsai@gmail.com), or hit me up on [Twitter](https://twitter.com/saasha_nair). See you soon! 😘

---

## Reading List

A list of resources to help gain a basic understanding of the field:

### General

1. [Privacy is Power: Why and How You Should Take Back Control of Your Data](https://www.carissaveliz.com/books) by Carrisa Véliz - Provides a non-technical view of the current privacy landscape and where we are headed if we continue along the current trajectory
2. Recommendations by Dr. Véliz

[https://twitter.com/carissaveliz/status/1334148170268483586?ref_src=twsrc^tfw](https://twitter.com/carissaveliz/status/1334148170268483586?ref_src=twsrc^tfw)

### Federated Learning

1. ["FL: Building better products with on-device data and privacy by default"](https://federated.withgoogle.com/) from Google AI - Fun non-tech online comic to help build an intuition
2. ["FL: Private Distributed ML"](https://mike.place/talks/fl/) by [Mike Lee Williams](https://twitter.com/mikepqr) - Engaging talk about the motivations, uses and challenges - [[Video](https://youtu.be/VUINeZUAlx8)]
3. [“Communication-Efficient Learning of Deep Networks from Decentralized Data”](https://arxiv.org/pdf/1602.05629.pdf) by McMahan et. al. - An easy to follow paper that explains FedAvg, which forms the basis of many currently used FL algos
4. [“Guarding user Privacy with FL and Differential Privacy”](https://youtu.be/e5othcNmync) by McMahan - Talk that summarizes the paper linked above; goes a step further to explain how Differential Privacy can be combined with FL

### Differential Privacy

1. [“DP: A Primer for a Non-tech Audience”](https://privacytools.seas.harvard.edu/files/privacytools/files/pedagogical-document-dp_0.pdf) by Nissim et. al. - As the name suggests, helps build an intuitive understanding of the topic
2. [“Why DP is awesome”](https://desfontain.es/privacy/differential-privacy-awesomeness.html) by [Damien Desfontaines](https://twitter.com/TedOnPrivacy) - Wonderfully rigorous series of posts that delve into the details of DP
3. [“Semi-supervised knowledge transfer for Deep Learning from Private Training Data”](https://arxiv.org/pdf/1610.05755.pdf) by Papernot et. al. - Introduces the PATE algorithm to perform DP with Deep Learning -
[[Blog post](http://www.cleverhans.io/privacy/2018/04/29/privacy-and-machine-learning.html)] [[Video](https://youtu.be/bDayquwDgjU)]

## Resources

Resources used in this post:

1. [During World War II, we did have something to hide](https://medium.com/@hansdezwart/during-world-war-ii-we-did-have-something-to-hide-40689565c550), by Hans de Zwart
2. [Privacy is Power: Why and How You Should Take Back Control of Your Data](https://www.carissaveliz.com/books) by Carrisa Véliz
3. [Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures](https://www.cs.cmu.edu/~mfredrik/papers/fjr2015ccs.pdf) by Fredrikson et. al.
4. [Privacy in Deep Learning: A Survey](https://arxiv.org/pdf/2004.12254.pdf) by Mireshghallah et. al.
5. ['Robust De-anonymization of Large Sparse Datasets'](https://www.cs.utexas.edu/~shmat/shmat_oak08netflix.pdf) by Narayanan et. al.
6. [Dr. Helen Nissenbaum's talk at PriCon2020](https://blog.openmined.org/conference-talk-summary-helen-nissenbaum-privacy-contextual-integrity-and-obfuscation/)
7. [Advances and Open Problems in Federated Learning](https://arxiv.org/pdf/1912.04977.pdf) by Kairouz et. al.
8. [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/pdf/1602.05629.pdf) by McMahan et. al.
9. [Differential Privacy: A Primer for a Non-technical Audience](https://privacytools.seas.harvard.edu/files/privacytools/files/pedagogical-document-dp_0.pdf) by Nissim et. al.
10. [Differential Privacy and Machine Learning: a Survey and Review](https://arxiv.org/pdf/1412.7584.pdf) by Ji et. al.
11. [Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data](https://arxiv.org/pdf/1610.05755.pdf) by Papernot et. al.
12. [Privacy and machine learning: two unexpected allies?](http://www.cleverhans.io/privacy/2018/04/29/privacy-and-machine-learning.html) by Papernot and Goodfellow