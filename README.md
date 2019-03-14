
# Variational Continual Learning

Original paper by **Cuong V. Nguyen**, **Yingzhen Li**, **Thang D. Bui** and **Richard E. Turner**

# Part 1. Paper Summary

## 1. Introduction

* **Continual Learning**

>* Data continuously arrive in a possibly non i.i.d. way
>* Tasks may change over time (e.g. new classes may be discovered)
>* Entirely new tasks can emerge ([Schlimmer & Fisher 1986](https://pdfs.semanticscholar.org/3883/850e09fec66f389697f90d84d468ed8aa062.pdf?_ga=2.88621278.4984534.1552575812-1435212181.1544760968); [Sutton & Whitehead, 1993](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.91.1872&rep=rep1&type=pdf); [Ring, 1997](https://link.springer.com/article/10.1023/A:1007331723572))

* **Challenge for Continual Learning**

>* Balance between **adapting to new data** vs. **retaining existing knowledge**
>* Too much plasticity $$\rightarrow$$ **catastrophic forgetting** ([McCloskey & Cohen, 1989](https://www.sciencedirect.com/science/article/pii/S0079742108605368); [Ratcliff, 1990](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.410.1393&rep=rep1&type=pdf); [Goodfellow et al., 2014a](https://arxiv.org/pdf/1312.6211.pdf))
>  * Too much stability $$\rightarrow$$ inability to adapt
>* **Approach 1:** train individual models on each task $\rightarrow$ train to combine them
>  * ([Lee et al., 2017](https://papers.nips.cc/paper/7051-overcoming-catastrophic-forgetting-by-incremental-moment-matching.pdf))
>* **Approach 2:** maintain a single model and use a single type of regularized training that prevents drastic changes in the influential parameters, but allow other parameters to change more freely
>  * ([Li & Hoiem, 2016](https://arxiv.org/pdf/1606.09282.pdf); [Kirkpatrick et al., 2017](https://arxiv.org/pdf/1612.00796.pdf); [Zenke et al., 2017](https://arxiv.org/pdf/1703.04200.pdf))

* **Variational Continual Learning**

>* Merge **online VI** ([Ghahramani & Attias, 2000](http://mlg.eng.cam.ac.uk/zoubin/papers/nips00w.pdf); [Sato, 2001](https://pdfs.semanticscholar.org/37a5/d49aa00999de4cd70a2ff3f8d3363892edae.pdf); [Broderick et al., 2013](https://papers.nips.cc/paper/4980-streaming-variational-bayes.pdf))
>* with **Monte Carlo VI for NN** ([Blundell et al., 2015](https://arxiv.org/pdf/1505.05424.pdf))
>* and include a **small episodic memory** ([Bachem et al., 2015](https://las.inf.ethz.ch/files/bachem15dpmeans.pdf); [Huggins et al., 2016](https://papers.nips.cc/paper/6486-coresets-for-scalable-bayesian-logistic-regression.pdf))

## 2. Continual Learning by Approximate Bayesian Inference

* **Online updating, derived from Bayes' rule**

>$$p(\boldsymbol{\theta}|\mathcal{D}_{1:T}) \propto p(\boldsymbol{\theta}) \prod^T_{t=1} \prod^{N_t}_{n_t=1} p(y_t^{(n_t)}|\boldsymbol{\theta},x_t^{(n_t)}) = p(\boldsymbol{\theta}) \prod^T_{t=1} p(\mathcal{D}_t|\boldsymbol{\theta}) \propto p(\boldsymbol{\theta}|\mathcal{D}_{1:T-1}) p(\mathcal{D}_T|\boldsymbol{\theta})$$

>* Posterior after $$T$$th dataset $$\propto$$ Posterior after $$(T-1)$$th dataset $$\times$$ Likelihood of the $$T$$th dataset

* **Projection Operation: approximation for intractable posterior** (recursive)

>$$\begin{align}
p(\boldsymbol{\theta}|\mathcal{D}_1) \approx q_1(\boldsymbol{\theta}) &= \text{proj}(p(\boldsymbol{\theta})p(\mathcal{D}_1|\boldsymbol{\theta})) \\
p(\boldsymbol{\theta}|\mathcal{D}_{1:T}) \approx q_T(\boldsymbol{\theta}) &= \text{proj}(q_{T-1}(\boldsymbol{\theta})p(\mathcal{D}_T|\boldsymbol{\theta}))
\end{align}$$

>|Projection Operation|Inference Method|References|
|-|-|-|
|Laplace's approximation    |Laplace propagation      |[Smola et al., 2004](references/Smola_2004.pdf)|
|Variational KL minimization|Online VI|[Ghahramani & Attias, 2000](references/Ghahramani&Attias_2000.pdf); [Sato, 2001](references/Sato_2001.pdf)|
|Moment matching            |Assumed density filtering|[Maybeck, 1982](references/Maybeck_1982.pdf)|
|Importance sampling        |Sequential Monte Carlo   |[Liu & Chen, 1998](references/Liu&Chen_1998.pdf)|

>* This paper will use **Online VI** as it outperforms other methods for complex models in the static setting ([Bui et al., 2016](references/Bui_2016.pdf))

### 2.1. VCL and Episodic Memory Enhancement

* **Projection Operation: KL Divergence Minimization**

>$$q_t(\boldsymbol{\theta}) = \underset{q \in \mathcal{Q}}{\text{argmin}} \text{KL}
\left( q(\boldsymbol{\theta}) || \frac{1}{Z_t} q_{t-1}(\boldsymbol{\theta}) p(\mathcal{D}_t|\boldsymbol{\theta}) \right)$$

>* $$q_0(\boldsymbol{\theta}) = p(\boldsymbol{\theta})$$
>* $$Z_t$$: normalizing constant (not required when computing the optimum)
>* VCL becomes Bayesian inference if $$p(\boldsymbol{\theta}|\mathcal{D}_{1:t}) \in \mathcal{Q} \;\forall\; t$$

* **Potential Problems**

>* Errors from repeated approximation $$\rightarrow$$ forget old tasks
>* Minimization at each step is also approximate $$\rightarrow$$ information loss

* **Solution: Coreset**

>* **Coreset:** small representative set of data from previously observed tasks
>  * Analogous to **episodic memory** ([Lopez-Paz & Ranzato, 2017](https://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning.pdf))
>* **Coreset VCL:** equivalent to a message-passing implementation of VI in which the coreset data point updates are scheduled after updating the other data
>* $$C_t$$: updated using $$C_{t-1}$$ and selected data points from $$\mathcal{D}_t$$ (e.g. random selection, K-center algorithm, ...)
>  * K-center algorithm: return K data points that are spread throughout the input space ([Gonzalez, 1985](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.366.8183&rep=rep1&type=pdf))

* **Variational Recursion**

>$$p(\boldsymbol{\theta}|\mathcal{D}_{1:t}) \propto p(\boldsymbol{\theta}|\mathcal{D}_{1:t} \setminus C_t) p(C_t|\boldsymbol{\theta}) \approx \tilde{q}_t (\boldsymbol{\theta}) p(C_t|\boldsymbol{\theta})$$

>$$p(\boldsymbol{\theta}|\mathcal{D}_{1:t} \setminus C_t) = p(\boldsymbol{\theta}|\mathcal{D}_{1:t-1} \setminus C_{t-1}) p(C_{t-1} \setminus C_t | \boldsymbol{\theta}) p(\mathcal{D}_t \setminus C_t | \boldsymbol{\theta}) \approx \tilde{q}_{t-1}(\boldsymbol{\theta}) p(\mathcal{D}_t \cup C_{t-1} \setminus C_t | \boldsymbol{\theta})$$

* **Algorithm**

>* **Step 1:** Observe $\mathcal{D}_t$
>* **Step 2:** Update $C_t$ using $C_{t-1}$ and $\mathcal{D}_t$
>* **Step 3:** Update $\tilde{q}_t$ (used for **propagation**)

>\begin{align}
\tilde{q}_t(\boldsymbol{\theta}) &= \text{proj} \left( \tilde{q}_{t-1}(\boldsymbol{\theta}) p(\mathcal{D}_t \cup C_{t-1} \setminus C_t | \boldsymbol{\theta}) \right) \\
&= \underset{q \in \mathcal{Q}}{\text{argmin}} \; \text{KL}
\left( q(\boldsymbol{\theta})  \;\big|\big|\; \frac{1}{\tilde{Z}} \tilde{q}_{t-1}(\boldsymbol{\theta}) p(\mathcal{D}_t \cup C_{t-1} \setminus C_t |\boldsymbol{\theta}) \right)
\end{align}

>* **Step 4:** Update $q_t$ (used for **prediction**)

>\begin{align}
q_t(\boldsymbol{\theta}) &= \text{proj} \left( \tilde{q}_{t}(\boldsymbol{\theta}) p(C_t | \boldsymbol{\theta}) \right) \\
&= \underset{q \in \mathcal{Q}}{\text{argmin}} \; \text{KL}
\left( q(\boldsymbol{\theta})  \;\big|\big|\; \frac{1}{Z} \tilde{q}_t (\boldsymbol{\theta}) p(C_t |\boldsymbol{\theta}) \right)
\end{align}

>* **Step 5:** Perform prediction

>$$p(y^*|\boldsymbol{x}^*, \mathcal{D}_{1:t}) = \int q_t(\boldsymbol{\theta}) p(y^*|\boldsymbol{\theta},\boldsymbol{x}^*) d\boldsymbol{\theta}$$

## 3. VCL in Deep Discriminative Models

* **Multi-head Networks**

>* Standard architecture used for multi-task learning ([Bakker & Heskes, 2003](http://www.jmlr.org/papers/volume4/bakker03a/bakker03a.pdf))
>* Share parameters close to the inputs / Separate heads for each output
>* **More advanced model structures:**
>  * for continual learning ([Rusu et al., 2016](https://arxiv.org/pdf/1606.04671.pdf))
>  * for multi-task learning in general ([Swietojanski & Renals, 2014](https://homepages.inf.ed.ac.uk/srenals/ps-slt14.pdf); [Rebuffi et al., 2017](http://www.robots.ox.ac.uk/~vedaldi/assets/pubs/rebuffi17learning.pdf))
>  * **automatic continual model building:** adding new structure as new tasks are encountered
>* This paper assumes that the model structure is known *a priori*

* **Formulation**

><img src = 'images/summary_1.png' width=350>

>* Model parameter $\boldsymbol{\theta} = \{ \boldsymbol{\theta}^H_{1:T}, \boldsymbol{\theta}^S \} \in \mathbb{R}^D$
>  * **Shared parameters:** updated constantly
>  * **Head parameter:** $q(\boldsymbol{\theta}^H_K) = p(\boldsymbol{\theta}^H_K)$ at the beginning, updated incrementally as each task emerges

>* For simplicity, use **Gaussian mean-field approximate posterior** $q_t(\boldsymbol{\theta}) = \prod^D_{d=1} \mathcal{N} (\theta_{t,d} ; \mu_{t,d}, \sigma^2_{t,d})$

* **Network Training**

>* Maximize the negative online variational free energy or the variational lower bound to the online marginal likelihood $\mathcal{L}^t_{VCL}$ with respect to the variational parameters $\{\mu_{t,d},\sigma_{t,d}\}^D_{d=1}$

>$$\mathcal{L}^t_{VCL} (q_t(\boldsymbol{\theta})) = \sum^{N_t}_{n=1} \mathbb{E}_{\boldsymbol{\theta} \sim q_t(\boldsymbol{\theta})} \left[ \log p(y_t^{(n)}|\boldsymbol{\theta},\mathbf{x}^{(n)}_t) \right] - \text{KL} (q_t(\boldsymbol{\theta})||q_{t-1}(\boldsymbol{\theta}))$$

>* $\text{KL} (q_t(\boldsymbol{\theta})||q_{t-1}(\boldsymbol{\theta}))$: tractable / set $q_0(\boldsymbol{\theta})$ as multivariate Gaussian ([Graves, 2011](https://www.cs.toronto.edu/~graves/nips_2011.pdf); [Blundell et al., 2015](https://arxiv.org/pdf/1505.05424.pdf))
>* $\mathbb{E}_{\boldsymbol{\theta} \sim q_t(\boldsymbol{\theta})} [\cdot]$: intractable $\rightarrow$ approximate by employing simple Monte Carlo and using the **local reparameterization trick** to compute the gradients ([Salimans & Knowles, 2013](https://arxiv.org/pdf/1206.6679.pdf); [Kingma & Welling, 2014](http://dpkingma.com/wordpress/wp-content/uploads/2014/10/iclr14_vae.pdf); [Kingma et al., 2015](https://arxiv.org/pdf/1506.02557.pdf))

## 4. VCL in Deep Generative Models

* **Deep Generative Models**

>* Can generate realistic images, sounds, and video sequences ([Chung et al., 2015](https://papers.nips.cc/paper/5653-a-recurrent-latent-variable-model-for-sequential-data.pdf); [Kingma et al., 2016](https://arxiv.org/pdf/1606.04934.pdf); [Vondrick et al., 2016](http://www.cs.columbia.edu/~vondrick/tinyvideo/paper.pdf))
>* Standard batch learning assumes observations to be i.i.d. and are all available at the same time
>* This paper applies VCL framework to **variational auto encoders** ([Kingma & Welling, 2014](http://dpkingma.com/wordpress/wp-content/uploads/2014/10/iclr14_vae.pdf); [Rezende et al., 2014](https://arxiv.org/pdf/1401.4082.pdf))

* **Formulation** - VAE approach (batch learning)

>$$p(\mathbf{x}|\mathbf{z},\boldsymbol{\theta}) p(\mathbf{z})$$

>* $p(\mathbf{z})$: prior over latent variables / typically Gaussian
>* $p(\mathbf{x}|\mathbf{z},\boldsymbol{\theta})$: defined by DNN, $\boldsymbol{f_\theta} (\mathbf{z})$, where $\boldsymbol{\theta}$ collects weight matrices and bias vectors
>* **Learning $\boldsymbol{\theta}$:** approximate MLE (maximize variational lower bound w.r.t. $\boldsymbol{\theta}$ and $\boldsymbol{\phi}$)

>$$\mathcal{L}_{\text{VAE}} (\boldsymbol{\theta},\boldsymbol{\phi}) = \sum^N_{n=1} \mathbb{E}_{q_\boldsymbol{\phi}(\mathbf{z}^{(n)}|\mathbf{x}^{(n)})}
\left[ \log \frac{p(\mathbf{x}^{(n)}|\mathbf{z}^{(n)},\boldsymbol{\theta})p(\mathbf{z}^{(n)})} {q_\boldsymbol{\phi} (\mathbf{z}^{(n)}|\mathbf{x}^{(n)})} \right]$$

>* $\rightarrow$ **No parameter uncertainty estimates** (used to weight the information learned from old data)

* **Formulation** - VCL approach (continual learning)

>* Approximate full posterior over parametrs, $q_t(\boldsymbol{\theta}) \approx p(\boldsymbol{\theta}|\mathcal{D}_{1:t})$
>* Maximize **full** variational lower bound w.r.t. $q_t$ and $\phi$

>$$\mathcal{L}^t_{\text{VAE}} (q_t(\boldsymbol{\theta}),\boldsymbol{\phi}) =
\mathbb{E}_{q_t(\boldsymbol{\theta})}\left\{
\sum^{N_t}_{n=1} \mathbb{E}_{q_\boldsymbol{\phi}(\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})}
\left[ \log \frac{p(\mathbf{x}_t^{(n)}|\mathbf{z}_t^{(n)},\boldsymbol{\theta})p(\mathbf{z}_t^{(n)})} {q_\boldsymbol{\phi} (\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})} \right] \right\}
-\text{KL}(q_t(\boldsymbol{\theta})||q_{t-1}(\boldsymbol{\theta}))$$

>* $\boldsymbol{\phi}$: task-specific $\rightarrow$ likely to be beneficial to share (parts of) these encoder networks

* **Model Architecture**

><img src = 'images/summary_2.png' width=350>

>* Latent variables $\mathbf{z}$ $\rightarrow$ Intermediate-level representations $\mathbf{h}$ $\rightarrow$ Observations $\mathbf{x}$

>* **Architecture 1:** shared bottom network - suitable when data are composed of a common set of structural primitives (e.g. strokes)
>* **Architecture 2:** shared head network - information tend to be entirely encoded in bottom network

## 5. Related Work

* **Continual Learning for Deep Discriminative Models** (regularized MLE)

>$$\mathcal{L}^t (\boldsymbol{\theta}) = \sum^{N_t}_{n=1} \log p(y_t^{(n)} | \boldsymbol{\theta},\mathbf{x}^{(n)}_t) - \frac{1}{2} \lambda_t (\boldsymbol{\theta} - \boldsymbol{\theta}_{t-1})^T \Sigma^{-1}_{t-1} (\boldsymbol{\theta} - \boldsymbol{\theta}_{t-1})$$

>* **ML Estimation** - set $\lambda_t = 0$

>* **MAP Estimation** - assume Gaussian prior $q(\boldsymbol{\theta}|\mathcal{D}_{1:t-1})=\mathcal{N}(\boldsymbol{\theta};\boldsymbol{\theta}_{t-1},\Sigma_{t-1}/\lambda_t)$
>  * $\Sigma_t=? \; \rightarrow \; \Sigma_t=I$ and use CV to find $\lambda_T$ $\rightarrow$ catastrophic forgetting

>* **Laplace Propagation (LP)** ([Smola et al., 2004](https://papers.nips.cc/paper/2444-laplace-propagation.pdf)) - recursion for $\Sigma_t$ using Laplace's approximation
>  * Diagonal LP: retain only the diagonal terms of $\Sigma^{-1}_t$ to avoid computing full Hessian

>$$\Sigma^{-1}_t = \Phi_t + \Sigma^{-1}_{t-1} \;\;\;,\;\;\; \Phi_t = - \nabla \nabla_\boldsymbol{\theta} \sum^{N_t}_{n=1} \log p(y^{(n)}_t | \boldsymbol{\theta}, \mathbf{x}^{(n)}_t) \big|_{\boldsymbol{\theta} = \boldsymbol{\theta}_t} \;\;\;,\;\;\; \lambda_t = 1$$

>* **Elastic Weight Consolidation (EWC)** ([Kirkpatrick et al., 2017](https://arxiv.org/pdf/1612.00796.pdf)) - modified diagonal LP
>  * Approximate the average Hessian of the likelihoods using Fisher information
>$$$$
>$$\Phi_t \approx \text{diag} \left( \sum^{N_t}_{n=1} \left( \nabla_\boldsymbol{\theta} \log p(y^{(n)}_t|\boldsymbol{\theta},\mathbf{x}^{(n)}_t) \right)^2 \;\Big|_{\boldsymbol{\theta}=\boldsymbol{\theta}_t}\right)$$
>$$$$
>  * Regularization term: introduce hyperparameter, remove prior, regularize intermediate estimates

>$$\frac{1}{2} (\boldsymbol{\theta} - \boldsymbol{\theta}_{t-1})^T (\Sigma^{-1}_0 + \Sigma^{t-1}_{t'=1} \Phi_{t'}) (\boldsymbol{\theta} - \boldsymbol{\theta}_{t-1})
\rightarrow
\frac{1}{2} \sum^{t-1}_{t'=1} \lambda_{t'} (\boldsymbol{\theta} - \boldsymbol{\theta}_{t'-1})^T \Phi_{t'} (\boldsymbol{\theta} - \boldsymbol{\theta}_{t'-1})$$

>* **Synaptic Intelligence (SI)** ([Zenke et al., 2017](https://arxiv.org/pdf/1703.04200.pdf)) - compute $\Sigma^{-1}_t$ using importance of each parameter to each task

* **Approximate Bayesian Training of NN** (focused on )

>|Approach|References|
|-|-|
|**extended Kalman filtering**|[Singhal & Wu, 1989](https://pdfs.semanticscholar.org/c4ea/370b7261f2a2bf3a9339cedb1ab1de348301.pdf?_ga=2.92679256.4984534.1552575812-1435212181.1544760968)|
|**Laplace's approximation**|[MacKay, 1992](https://authors.library.caltech.edu/13793/1/MACnc92b.pdf)|
|**variational inference**|[Hinton & Van Camp, 1993](https://www.cs.toronto.edu/~hinton/absps/colt93.pdf); [Barber & Bishop, 1998](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-ensemble-nato-98.pdf); [Graves, 2011](https://www.cs.toronto.edu/~graves/nips_2011.pdf); [Blundell et al., 2015](https://arxiv.org/pdf/1505.05424.pdf); [Gal & Ghahramani, 2016](https://arxiv.org/pdf/1506.02142.pdf)|
|**sequential Monte Carlo**|[de Freitas et al., 2000](https://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015664)|
|**expectation propagation**|[Hernández-Lobato & Adams, 2015](https://arxiv.org/pdf/1502.05336.pdf)|
|**approximate power EP**|[Hernández-Lobato et al., 2016](https://arxiv.org/pdf/1511.03243.pdf)|

* **Continual Learning for Deep Generative Models**

>* **Naïve approach:** apply VAE to $\mathcal{D}_t$ with parameters initialized at $\boldsymbol{\theta}_{t-1}$ $\rightarrow$ catastrophic forgetting
>* **Alternative:** add EWC regularization term to VAE objective & approximate marginal likelihood by variational lower bound
>  * Similar approximation can be used for **Hessian matrices for LP** and **$\Sigma^{-1}_t$ for SI** (Importance sampling: [Burda et al., 2016](https://arxiv.org/pdf/1509.00519.pdf))

>$$\mathcal{L}^t_{EWC}
(\boldsymbol{\theta},\boldsymbol{\phi}) = \sum^{N_t}_{n=1} \mathbb{E}_{q_\boldsymbol{\phi}(\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})}
\left[ \log \frac{p(\mathbf{x}_t^{(n)}|\mathbf{z}_t^{(n)},\boldsymbol{\theta})p(\mathbf{z}_t^{(n)})} {q_\boldsymbol{\phi} (\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})} \right]
- \frac{1}{2} \sum^{t-1}_{t'=1} \lambda_{t'} (\boldsymbol{\theta} - \boldsymbol{\theta}_{t'-1})^T \Phi_{t'} (\boldsymbol{\theta} - \boldsymbol{\theta}_{t'-1})$$

>$$\Phi_t \approx \text{diag} \left( \sum^{N_t}_{n=1} \left( \nabla_\boldsymbol{\theta}
\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})}
\left[ \log \frac{p(\mathbf{x}_t^{(n)}|\mathbf{z}_t^{(n)},\boldsymbol{\theta})p(\mathbf{z}_t^{(n)})} {q_\boldsymbol{\phi} (\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})} \right]
\right)^2 \;\Bigg|_{\boldsymbol{\theta}=\boldsymbol{\theta}_t}\right)$$


## 6. Experiments

### 6.1. Experiments with Deep Discriminative Models

* **Permutated MNIST** (task-specific random permutation)

>* **Experiment Setup** (same for all models)

>|||
|-|-|
|**Architecture**|FC single-head, 2 hidden layers $\times$ 100 units, ReLU|
|**Optimizer**|Adam with $lr=10^{-3}$|
|**Metric**|Test set accuracy on all "observed" tasks|

>* **Experiment Setup** (model-specific)
>  * **EWC** - trained w/o dropout, Fisher info. matrices approximated using 600 random samples from current dataset
>  * **Diagonal LP** - trained with prior $\mathcal{N}(0,\mathbf{I})$, Hessian of LP approximated using Fisher info. matrices with 200 samples

>|                |VCL|Coreset Only|SI|EWC|Diagonal LP|
|----------------|-|-|-|-|-|
|**Hyper-parameters**|N/A|N/A|$0.01,0.1,\mathbf{0.5},1,2$|$1,10,\mathbf{10^2},10^3,10^4$|$0.01,\mathbf{0.1},1,10,100$|
|**Batch Size / Epochs**|256 / 100<br/>coreset = 200|200 / 100<br/>(VFE method)|256 / 20|200 / 20|200 / 20|

>* **Experiment 1-1: Learning Methods**
>  * VCL (no coreset, random coreset, coreset selected using K-center method) vs. EWC, SI, diagonal LP
>  

><img src = 'images/summary_3.png' width=600>

>* **Experiment 1-2: Coreset Size**
>  * Performance improves with coreset size & converges (large coreset is fully representative of the task)

><img src = 'images/summary_4.png' width=600>

* **Split MNIST**

>* **Experiment Setup** (same for all models)

>|||
|-|-|
|**Architecture**|FC multi-head, 2 hidden layers $\times$ 256 units, ReLU|
|**Optimizer**|Adam with $lr=10^{-3}$|
|**Initialization**|Use prior $\mathcal{N}(0,\mathbf{I})$<br/>initialize optimizer at **mean of ML model** and **small variance** ($10^{-6}$)|
|**Metric**|Test set accuracy of the current model on all observed tasks separately<br/>& Average accuracy over all tasks|
|**Random Seed**|10 runs with different seed averaged|
|$q_t(\boldsymbol{\theta})$|computed for each task separately using coreset points corresponding to the task|

>* **Experiment Setup** (model-specific)
>  * **EWC** - Fisher info. matrices approximated using 200 random samples from current dataset
>  * **EWC** - single-head vs. multi-head $\rightarrow$ performance insensitive to $\lambda$ & multi-head is better

>  * **Diagonal LP** - prior $\mathcal{N}(0,\mathbf{I})$, Hessian approximated using Fisher info. matrices with 200 samples

>|                |VCL|Coreset Only|SI|EWC|Diagonal LP|
|----------------|-|-|-|-|-|
|**Hyper-parameters**|N/A|N/A|$0.01,0.1,\mathbf{1},2,3$|$\mathbf{1},10,10^2,10^3,10^4$|$\mathbf{1}$|
|**Batch Size / Epochs**|training set size / 120<br/>coreset = 40|coreset size / 120<br/>(VFE method)||||

>* **Results**

><img src = 'images/summary_5.png' width=600>

* **Split notMNIST**

>* **Experiment Setup** same as Split MNIST except:
>  * Deeper Architecture: 4 hidden layers $\times$ 150 units
>  * $\lambda$: $0.1$ for SI, $10^4$ for multi-head EWC, $1$ for multi-head LP

>* **Results**

><img src = 'images/summary_6.png' width=600>

### 6.2. Experiments with Deep Generative Models

* **MNIST Digit Generation** and **notMNIST Character Generation**

>* **Experiment Setup**
>  * lr: $10^{-4}$
>  * epochs: 200 for MNIST (400 for SI) / 400 for notMNIST
>  * VCL: $q_t(\boldsymbol{\theta})$ initialization - mean same as $q_{t-1}(\boldsymbol{\theta})$ and $\log \sigma = 10^{-6}$

>|||
|-|-|
|**Architecture**|1 hidden layer $\times$ 500 units (both shared & task-specific)<br/>$\mathbf{h}$=500-dim, $\mathbf{z}$=50-dim|
|**Dataset**|10 datasets received in sequence|
|**Metrics**|Importance sampling estimate of test-LL using 5,000 samples<br/>classifier uncertainty (KL-divergence)|
|**Models**|VCL, naive online learning using VAE objective, LP, EWC and SI|

>* **Results**

><img src = 'images/summary_7.png' width=600>

><img src = 'images/summary_8.png' width=600>

><img src = 'images/summary_9.png' width=600>
