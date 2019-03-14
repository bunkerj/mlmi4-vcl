
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
>* Too much plasticity → **catastrophic forgetting** ([McCloskey & Cohen, 1989](https://www.sciencedirect.com/science/article/pii/S0079742108605368); [Ratcliff, 1990](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.410.1393&rep=rep1&type=pdf); [Goodfellow et al., 2014a](https://arxiv.org/pdf/1312.6211.pdf))
>  * Too much stability → inability to adapt
>* **Approach 1:** train individual models on each task → train to combine them
>  * ([Lee et al., 2017](https://papers.nips.cc/paper/7051-overcoming-catastrophic-forgetting-by-incremental-moment-matching.pdf))
>* **Approach 2:** maintain a single model and use a single type of regularized training that prevents drastic changes in the influential parameters, but allow other parameters to change more freely
>  * ([Li & Hoiem, 2016](https://arxiv.org/pdf/1606.09282.pdf); [Kirkpatrick et al., 2017](https://arxiv.org/pdf/1612.00796.pdf); [Zenke et al., 2017](https://arxiv.org/pdf/1703.04200.pdf))

* **Variational Continual Learning**

>* Merge **online VI** ([Ghahramani & Attias, 2000](http://mlg.eng.cam.ac.uk/zoubin/papers/nips00w.pdf); [Sato, 2001](https://pdfs.semanticscholar.org/37a5/d49aa00999de4cd70a2ff3f8d3363892edae.pdf); [Broderick et al., 2013](https://papers.nips.cc/paper/4980-streaming-variational-bayes.pdf))
>* with **Monte Carlo VI for NN** ([Blundell et al., 2015](https://arxiv.org/pdf/1505.05424.pdf))
>* and include a **small episodic memory** ([Bachem et al., 2015](https://las.inf.ethz.ch/files/bachem15dpmeans.pdf); [Huggins et al., 2016](https://papers.nips.cc/paper/6486-coresets-for-scalable-bayesian-logistic-regression.pdf))

## 2. Continual Learning by Approximate Bayesian Inference

* **Online updating, derived from Bayes' rule**

><a href="https://www.codecogs.com/eqnedit.php?latex=$$p(\boldsymbol{\theta}|\mathcal{D}_{1:T})&space;\propto&space;p(\boldsymbol{\theta})&space;\prod^T_{t=1}&space;\prod^{N_t}_{n_t=1}&space;p(y_t^{(n_t)}|\boldsymbol{\theta},x_t^{(n_t)})&space;=&space;p(\boldsymbol{\theta})&space;\prod^T_{t=1}&space;p(\mathcal{D}_t|\boldsymbol{\theta})&space;\propto&space;p(\boldsymbol{\theta}|\mathcal{D}_{1:T-1})&space;p(\mathcal{D}_T|\boldsymbol{\theta})$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$p(\boldsymbol{\theta}|\mathcal{D}_{1:T})&space;\propto&space;p(\boldsymbol{\theta})&space;\prod^T_{t=1}&space;\prod^{N_t}_{n_t=1}&space;p(y_t^{(n_t)}|\boldsymbol{\theta},x_t^{(n_t)})&space;=&space;p(\boldsymbol{\theta})&space;\prod^T_{t=1}&space;p(\mathcal{D}_t|\boldsymbol{\theta})&space;\propto&space;p(\boldsymbol{\theta}|\mathcal{D}_{1:T-1})&space;p(\mathcal{D}_T|\boldsymbol{\theta})$$" title="$$p(\boldsymbol{\theta}|\mathcal{D}_{1:T}) \propto p(\boldsymbol{\theta}) \prod^T_{t=1} \prod^{N_t}_{n_t=1} p(y_t^{(n_t)}|\boldsymbol{\theta},x_t^{(n_t)}) = p(\boldsymbol{\theta}) \prod^T_{t=1} p(\mathcal{D}_t|\boldsymbol{\theta}) \propto p(\boldsymbol{\theta}|\mathcal{D}_{1:T-1}) p(\mathcal{D}_T|\boldsymbol{\theta})$$" /></a>

>* Posterior after Tth dataset is proportional to the Posterior after the (T-1)th dataset multiplied by the Likelihood of the Tth dataset

* **Projection Operation: approximation for intractable posterior** (recursive)

><a href="https://www.codecogs.com/eqnedit.php?latex=$$\begin{align*}&space;p(\boldsymbol{\theta}|\mathcal{D}_1)&space;\approx&space;q_1(\boldsymbol{\theta})&space;=&space;\text{proj}(p(\boldsymbol{\theta})p(\mathcal{D}1|\boldsymbol{\theta}))&space;\&space;p(\boldsymbol{\theta}|\mathcal{D}{1:T})&space;\approx&space;q_T(\boldsymbol{\theta})&space;&=&space;\text{proj}(q_{T-1}(\boldsymbol{\theta})p(\mathcal{D}_T|\boldsymbol{\theta}))&space;\end{align}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\begin{align*}&space;p(\boldsymbol{\theta}|\mathcal{D}_1)&space;\approx&space;q_1(\boldsymbol{\theta})&space;=&space;\text{proj}(p(\boldsymbol{\theta})p(\mathcal{D}1|\boldsymbol{\theta}))&space;\&space;p(\boldsymbol{\theta}|\mathcal{D}{1:T})&space;\approx&space;q_T(\boldsymbol{\theta})&space;&=&space;\text{proj}(q_{T-1}(\boldsymbol{\theta})p(\mathcal{D}_T|\boldsymbol{\theta}))&space;\end{align}$$" title="$$\begin{align*} p(\boldsymbol{\theta}|\mathcal{D}_1) \approx q_1(\boldsymbol{\theta}) = \text{proj}(p(\boldsymbol{\theta})p(\mathcal{D}1|\boldsymbol{\theta})) \ p(\boldsymbol{\theta}|\mathcal{D}{1:T}) \approx q_T(\boldsymbol{\theta}) &= \text{proj}(q_{T-1}(\boldsymbol{\theta})p(\mathcal{D}_T|\boldsymbol{\theta})) \end{align}$$" /></a>

>|Projection Operation|Inference Method|References|
|-|-|-|
|Laplace's approximation    |Laplace propagation      |[Smola et al., 2004](references/Smola_2004.pdf)|
|Variational KL minimization|Online VI|[Ghahramani & Attias, 2000](references/Ghahramani&Attias_2000.pdf); [Sato, 2001](references/Sato_2001.pdf)|
|Moment matching            |Assumed density filtering|[Maybeck, 1982](references/Maybeck_1982.pdf)|
|Importance sampling        |Sequential Monte Carlo   |[Liu & Chen, 1998](references/Liu&Chen_1998.pdf)|

>* This paper will use **Online VI** as it outperforms other methods for complex models in the static setting ([Bui et al., 2016](references/Bui_2016.pdf))

### 2.1. VCL and Episodic Memory Enhancement

* **Projection Operation: KL Divergence Minimization**

><a href="https://www.codecogs.com/eqnedit.php?latex=$$q_t(\boldsymbol{\theta})&space;=&space;\underset{q&space;\in&space;\mathcal{Q}}{\text{argmin}}&space;\text{KL}&space;\left(&space;q(\boldsymbol{\theta})&space;||&space;\frac{1}{Z_t}&space;q_{t-1}(\boldsymbol{\theta})&space;p(\mathcal{D}_t|\boldsymbol{\theta})&space;\right)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$q_t(\boldsymbol{\theta})&space;=&space;\underset{q&space;\in&space;\mathcal{Q}}{\text{argmin}}&space;\text{KL}&space;\left(&space;q(\boldsymbol{\theta})&space;||&space;\frac{1}{Z_t}&space;q_{t-1}(\boldsymbol{\theta})&space;p(\mathcal{D}_t|\boldsymbol{\theta})&space;\right)$$" title="$$q_t(\boldsymbol{\theta}) = \underset{q \in \mathcal{Q}}{\text{argmin}} \text{KL} \left( q(\boldsymbol{\theta}) || \frac{1}{Z_t} q_{t-1}(\boldsymbol{\theta}) p(\mathcal{D}_t|\boldsymbol{\theta}) \right)$$" /></a>

>* <a href="https://www.codecogs.com/eqnedit.php?latex=$$q_0(\boldsymbol{\theta})&space;=&space;p(\boldsymbol{\theta})$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$q_0(\boldsymbol{\theta})&space;=&space;p(\boldsymbol{\theta})$$" title="$$q_0(\boldsymbol{\theta}) = p(\boldsymbol{\theta})$$" /></a>
>* <a href="https://www.codecogs.com/eqnedit.php?latex=$$Z_t$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$Z_t$$" title="$$Z_t$$" /></a>: normalizing constant (not required when computing the optimum)
>* VCL becomes Bayesian inference if <a href="https://www.codecogs.com/eqnedit.php?latex=$$p(\boldsymbol{\theta}|\mathcal{D}_{1:t})&space;\in&space;\mathcal{Q}&space;\;\forall\;&space;t$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$p(\boldsymbol{\theta}|\mathcal{D}_{1:t})&space;\in&space;\mathcal{Q}&space;\;\forall\;&space;t$$" title="$$p(\boldsymbol{\theta}|\mathcal{D}_{1:t}) \in \mathcal{Q} \;\forall\; t$$" /></a>

* **Potential Problems**

>* Errors from repeated approximation → forget old tasks
>* Minimization at each step is also approximate → information loss

* **Solution: Coreset**

>* **Coreset:** small representative set of data from previously observed tasks
>  * Analogous to **episodic memory** ([Lopez-Paz & Ranzato, 2017](https://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning.pdf))
>* **Coreset VCL:** equivalent to a message-passing implementation of VI in which the coreset data point updates are scheduled after updating the other data
>* $$C_t$$: updated using $$C_{t-1}$$ and selected data points from $$\mathcal{D}_t$$ (e.g. random selection, K-center algorithm, ...)
>  * K-center algorithm: return K data points that are spread throughout the input space ([Gonzalez, 1985](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.366.8183&rep=rep1&type=pdf))

* **Variational Recursion**

><a href="https://www.codecogs.com/eqnedit.php?latex=$$p(\boldsymbol{\theta}|\mathcal{D}_{1:t})&space;\propto&space;p(\boldsymbol{\theta}|\mathcal{D}_{1:t}&space;\setminus&space;C_t)&space;p(C_t|\boldsymbol{\theta})&space;\approx&space;\tilde{q}_t&space;(\boldsymbol{\theta})&space;p(C_t|\boldsymbol{\theta})$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$p(\boldsymbol{\theta}|\mathcal{D}_{1:t})&space;\propto&space;p(\boldsymbol{\theta}|\mathcal{D}_{1:t}&space;\setminus&space;C_t)&space;p(C_t|\boldsymbol{\theta})&space;\approx&space;\tilde{q}_t&space;(\boldsymbol{\theta})&space;p(C_t|\boldsymbol{\theta})$$" title="$$p(\boldsymbol{\theta}|\mathcal{D}_{1:t}) \propto p(\boldsymbol{\theta}|\mathcal{D}_{1:t} \setminus C_t) p(C_t|\boldsymbol{\theta}) \approx \tilde{q}_t (\boldsymbol{\theta}) p(C_t|\boldsymbol{\theta})$$" /></a>


><a href="https://www.codecogs.com/eqnedit.php?latex=$$p(\boldsymbol{\theta}|\mathcal{D}_{1:t}&space;\setminus&space;C_t)&space;=&space;p(\boldsymbol{\theta}|\mathcal{D}_{1:t-1}&space;\setminus&space;C_{t-1})&space;p(C_{t-1}&space;\setminus&space;C_t&space;|&space;\boldsymbol{\theta})&space;p(\mathcal{D}_t&space;\setminus&space;C_t&space;|&space;\boldsymbol{\theta})&space;\approx&space;\tilde{q}_{t-1}(\boldsymbol{\theta})&space;p(\mathcal{D}_t&space;\cup&space;C_{t-1}&space;\setminus&space;C_t&space;|&space;\boldsymbol{\theta})$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$p(\boldsymbol{\theta}|\mathcal{D}_{1:t}&space;\setminus&space;C_t)&space;=&space;p(\boldsymbol{\theta}|\mathcal{D}_{1:t-1}&space;\setminus&space;C_{t-1})&space;p(C_{t-1}&space;\setminus&space;C_t&space;|&space;\boldsymbol{\theta})&space;p(\mathcal{D}_t&space;\setminus&space;C_t&space;|&space;\boldsymbol{\theta})&space;\approx&space;\tilde{q}_{t-1}(\boldsymbol{\theta})&space;p(\mathcal{D}_t&space;\cup&space;C_{t-1}&space;\setminus&space;C_t&space;|&space;\boldsymbol{\theta})$$" title="$$p(\boldsymbol{\theta}|\mathcal{D}_{1:t} \setminus C_t) = p(\boldsymbol{\theta}|\mathcal{D}_{1:t-1} \setminus C_{t-1}) p(C_{t-1} \setminus C_t | \boldsymbol{\theta}) p(\mathcal{D}_t \setminus C_t | \boldsymbol{\theta}) \approx \tilde{q}_{t-1}(\boldsymbol{\theta}) p(\mathcal{D}_t \cup C_{t-1} \setminus C_t | \boldsymbol{\theta})$$" /></a>

* **Algorithm**

>* **Step 1:** Observe $\mathcal{D}_t$
>* **Step 2:** Update $C_t$ using $C_{t-1}$ and $\mathcal{D}_t$
>* **Step 3:** Update $\tilde{q}_t$ (used for **propagation**)

><a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;\tilde{q}_t(\boldsymbol{\theta})&space;&=&space;\text{proj}&space;\left(&space;\tilde{q}_{t-1}(\boldsymbol{\theta})&space;p(\mathcal{D}_t&space;\cup&space;C_{t-1}&space;\setminus&space;C_t&space;|&space;\boldsymbol{\theta})&space;\right)&space;\\&space;&=&space;\underset{q&space;\in&space;\mathcal{Q}}{\text{argmin}}&space;\;&space;\text{KL}&space;\left(&space;q(\boldsymbol{\theta})&space;\;\big|\big|\;&space;\frac{1}{\tilde{Z}}&space;\tilde{q}_{t-1}(\boldsymbol{\theta})&space;p(\mathcal{D}_t&space;\cup&space;C_{t-1}&space;\setminus&space;C_t&space;|\boldsymbol{\theta})&space;\right)&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\tilde{q}_t(\boldsymbol{\theta})&space;&=&space;\text{proj}&space;\left(&space;\tilde{q}_{t-1}(\boldsymbol{\theta})&space;p(\mathcal{D}_t&space;\cup&space;C_{t-1}&space;\setminus&space;C_t&space;|&space;\boldsymbol{\theta})&space;\right)&space;\\&space;&=&space;\underset{q&space;\in&space;\mathcal{Q}}{\text{argmin}}&space;\;&space;\text{KL}&space;\left(&space;q(\boldsymbol{\theta})&space;\;\big|\big|\;&space;\frac{1}{\tilde{Z}}&space;\tilde{q}_{t-1}(\boldsymbol{\theta})&space;p(\mathcal{D}_t&space;\cup&space;C_{t-1}&space;\setminus&space;C_t&space;|\boldsymbol{\theta})&space;\right)&space;\end{align*}" title="\begin{align*} \tilde{q}_t(\boldsymbol{\theta}) &= \text{proj} \left( \tilde{q}_{t-1}(\boldsymbol{\theta}) p(\mathcal{D}_t \cup C_{t-1} \setminus C_t | \boldsymbol{\theta}) \right) \\ &= \underset{q \in \mathcal{Q}}{\text{argmin}} \; \text{KL} \left( q(\boldsymbol{\theta}) \;\big|\big|\; \frac{1}{\tilde{Z}} \tilde{q}_{t-1}(\boldsymbol{\theta}) p(\mathcal{D}_t \cup C_{t-1} \setminus C_t |\boldsymbol{\theta}) \right) \end{align*}" /></a>

>* **Step 4:** Update $q_t$ (used for **prediction**)

><a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;q_t(\boldsymbol{\theta})&space;&=&space;\text{proj}&space;\left(&space;\tilde{q}_{t}(\boldsymbol{\theta})&space;p(C_t&space;|&space;\boldsymbol{\theta})&space;\right)&space;\\&space;&=&space;\underset{q&space;\in&space;\mathcal{Q}}{\text{argmin}}&space;\;&space;\text{KL}&space;\left(&space;q(\boldsymbol{\theta})&space;\;\big|\big|\;&space;\frac{1}{Z}&space;\tilde{q}_t&space;(\boldsymbol{\theta})&space;p(C_t&space;|\boldsymbol{\theta})&space;\right)&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;q_t(\boldsymbol{\theta})&space;&=&space;\text{proj}&space;\left(&space;\tilde{q}_{t}(\boldsymbol{\theta})&space;p(C_t&space;|&space;\boldsymbol{\theta})&space;\right)&space;\\&space;&=&space;\underset{q&space;\in&space;\mathcal{Q}}{\text{argmin}}&space;\;&space;\text{KL}&space;\left(&space;q(\boldsymbol{\theta})&space;\;\big|\big|\;&space;\frac{1}{Z}&space;\tilde{q}_t&space;(\boldsymbol{\theta})&space;p(C_t&space;|\boldsymbol{\theta})&space;\right)&space;\end{align*}" title="\begin{align*} q_t(\boldsymbol{\theta}) &= \text{proj} \left( \tilde{q}_{t}(\boldsymbol{\theta}) p(C_t | \boldsymbol{\theta}) \right) \\ &= \underset{q \in \mathcal{Q}}{\text{argmin}} \; \text{KL} \left( q(\boldsymbol{\theta}) \;\big|\big|\; \frac{1}{Z} \tilde{q}_t (\boldsymbol{\theta}) p(C_t |\boldsymbol{\theta}) \right) \end{align*}" /></a>

>* **Step 5:** Perform prediction

><a href="https://www.codecogs.com/eqnedit.php?latex=$$p(y^*|\boldsymbol{x}^*,&space;\mathcal{D}_{1:t})&space;=&space;\int&space;q_t(\boldsymbol{\theta})&space;p(y^*|\boldsymbol{\theta},\boldsymbol{x}^*)&space;d\boldsymbol{\theta}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$p(y^*|\boldsymbol{x}^*,&space;\mathcal{D}_{1:t})&space;=&space;\int&space;q_t(\boldsymbol{\theta})&space;p(y^*|\boldsymbol{\theta},\boldsymbol{x}^*)&space;d\boldsymbol{\theta}$$" title="$$p(y^*|\boldsymbol{x}^*, \mathcal{D}_{1:t}) = \int q_t(\boldsymbol{\theta}) p(y^*|\boldsymbol{\theta},\boldsymbol{x}^*) d\boldsymbol{\theta}$$" /></a>

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

>* Model parameter <a href="https://www.codecogs.com/eqnedit.php?latex=$\bm{\theta}&space;=&space;\{&space;\bm{\theta}^H_{1:T},&space;\bm{\theta}^S&space;\}&space;\in&space;\mathbb{R}^D$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\bm{\theta}&space;=&space;\{&space;\bm{\theta}^H_{1:T},&space;\bm{\theta}^S&space;\}&space;\in&space;\mathbb{R}^D$" title="$\bm{\theta} = \{ \bm{\theta}^H_{1:T}, \bm{\theta}^S \} \in \mathbb{R}^D$" /></a>
>  * **Shared parameters:** updated constantly
>  * **Head parameter:** <a href="https://www.codecogs.com/eqnedit.php?latex=$q(\bm{\theta}^H_K)&space;=&space;p(\bm{\theta}^H_K)$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$q(\bm{\theta}^H_K)&space;=&space;p(\bm{\theta}^H_K)$" title="$q(\bm{\theta}^H_K) = p(\bm{\theta}^H_K)$" /></a> at the beginning, updated incrementally as each task emerges

>* For simplicity, use **Gaussian mean-field approximate posterior** <a href="https://www.codecogs.com/eqnedit.php?latex=$q_t(\bm{\theta})&space;=&space;\prod^D_{d=1}&space;\mathcal{N}&space;(\theta_{t,d}&space;;&space;\mu_{t,d},&space;\sigma^2_{t,d})$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$q_t(\bm{\theta})&space;=&space;\prod^D_{d=1}&space;\mathcal{N}&space;(\theta_{t,d}&space;;&space;\mu_{t,d},&space;\sigma^2_{t,d})$" title="$q_t(\bm{\theta}) = \prod^D_{d=1} \mathcal{N} (\theta_{t,d} ; \mu_{t,d}, \sigma^2_{t,d})$" /></a>

* **Network Training**

>* Maximize the negative online variational free energy or the variational lower bound to the online marginal likelihood <a href="https://www.codecogs.com/eqnedit.php?latex=$\mathcal{L}^t_{VCL}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\mathcal{L}^t_{VCL}$" title="$\mathcal{L}^t_{VCL}$" /></a> with respect to the variational parameters <a href="https://www.codecogs.com/eqnedit.php?latex=$\{\mu_{t,d},\sigma_{t,d}\}^D_{d=1}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\{\mu_{t,d},\sigma_{t,d}\}^D_{d=1}$" title="$\{\mu_{t,d},\sigma_{t,d}\}^D_{d=1}$" /></a>

><a href="https://www.codecogs.com/eqnedit.php?latex=$$\mathcal{L}^t_{VCL}&space;(q_t(\boldsymbol{\theta}))&space;=&space;\sum^{N_t}_{n=1}&space;\mathbb{E}_{\boldsymbol{\theta}&space;\sim&space;q_t(\boldsymbol{\theta})}&space;\left[&space;\log&space;p(y_t^{(n)}|\boldsymbol{\theta},\mathbf{x}^{(n)}_t)&space;\right]&space;-&space;\text{KL}&space;(q_t(\boldsymbol{\theta})||q_{t-1}(\boldsymbol{\theta}))$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\mathcal{L}^t_{VCL}&space;(q_t(\boldsymbol{\theta}))&space;=&space;\sum^{N_t}_{n=1}&space;\mathbb{E}_{\boldsymbol{\theta}&space;\sim&space;q_t(\boldsymbol{\theta})}&space;\left[&space;\log&space;p(y_t^{(n)}|\boldsymbol{\theta},\mathbf{x}^{(n)}_t)&space;\right]&space;-&space;\text{KL}&space;(q_t(\boldsymbol{\theta})||q_{t-1}(\boldsymbol{\theta}))$$" title="$$\mathcal{L}^t_{VCL} (q_t(\boldsymbol{\theta})) = \sum^{N_t}_{n=1} \mathbb{E}_{\boldsymbol{\theta} \sim q_t(\boldsymbol{\theta})} \left[ \log p(y_t^{(n)}|\boldsymbol{\theta},\mathbf{x}^{(n)}_t) \right] - \text{KL} (q_t(\boldsymbol{\theta})||q_{t-1}(\boldsymbol{\theta}))$$" /></a>

>* <a href="https://www.codecogs.com/eqnedit.php?latex=$\text{KL}&space;(q_t(\bm{\theta})||q_{t-1}(\bm{\theta}))$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\text{KL}&space;(q_t(\bm{\theta})||q_{t-1}(\bm{\theta}))$" title="$\text{KL} (q_t(\bm{\theta})||q_{t-1}(\bm{\theta}))$" /></a>: tractable / set $q_0(\boldsymbol{\theta})$ as multivariate Gaussian ([Graves, 2011](https://www.cs.toronto.edu/~graves/nips_2011.pdf); [Blundell et al., 2015](https://arxiv.org/pdf/1505.05424.pdf))
>* <a href="https://www.codecogs.com/eqnedit.php?latex=$\mathbb{E}_{\bm{\theta}&space;\sim&space;q_t(\bm{\theta})}&space;[\cdot]$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\mathbb{E}_{\bm{\theta}&space;\sim&space;q_t(\bm{\theta})}&space;[\cdot]$" title="$\mathbb{E}_{\bm{\theta} \sim q_t(\bm{\theta})} [\cdot]$" /></a>: intractable → approximate by employing simple Monte Carlo and using the **local reparameterization trick** to compute the gradients ([Salimans & Knowles, 2013](https://arxiv.org/pdf/1206.6679.pdf); [Kingma & Welling, 2014](http://dpkingma.com/wordpress/wp-content/uploads/2014/10/iclr14_vae.pdf); [Kingma et al., 2015](https://arxiv.org/pdf/1506.02557.pdf))

## 4. VCL in Deep Generative Models

* **Deep Generative Models**

>* Can generate realistic images, sounds, and video sequences ([Chung et al., 2015](https://papers.nips.cc/paper/5653-a-recurrent-latent-variable-model-for-sequential-data.pdf); [Kingma et al., 2016](https://arxiv.org/pdf/1606.04934.pdf); [Vondrick et al., 2016](http://www.cs.columbia.edu/~vondrick/tinyvideo/paper.pdf))
>* Standard batch learning assumes observations to be i.i.d. and are all available at the same time
>* This paper applies VCL framework to **variational auto encoders** ([Kingma & Welling, 2014](http://dpkingma.com/wordpress/wp-content/uploads/2014/10/iclr14_vae.pdf); [Rezende et al., 2014](https://arxiv.org/pdf/1401.4082.pdf))

* **Formulation** - VAE approach (batch learning)

><a href="https://www.codecogs.com/eqnedit.php?latex=$$p(\mathbf{x}|\mathbf{z},\boldsymbol{\theta})&space;p(\mathbf{z})$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$p(\mathbf{x}|\mathbf{z},\boldsymbol{\theta})&space;p(\mathbf{z})$$" title="$$p(\mathbf{x}|\mathbf{z},\boldsymbol{\theta}) p(\mathbf{z})$$" /></a>

>* <a href="https://www.codecogs.com/eqnedit.php?latex=$p(\mathbf{z})$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$p(\mathbf{z})$" title="$p(\mathbf{z})$" /></a>: prior over latent variables / typically Gaussian
>* <a href="https://www.codecogs.com/eqnedit.php?latex=$p(\mathbf{x}\lvert\mathbf{z},\mathbf{\theta})$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$p(\mathbf{x}\lvert\mathbf{z},\mathbf{\theta})$" title="$p(\mathbf{x}\lvert\mathbf{z},\mathbf{\theta})$" /></a>: defined by DNN, <a href="https://www.codecogs.com/eqnedit.php?latex=$\mathbf{f_\theta}&space;(\mathbf{z})$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\mathbf{f_\theta}&space;(\mathbf{z})$" title="$\mathbf{f_\theta} (\mathbf{z})$" /></a>, where <a href="https://www.codecogs.com/eqnedit.php?latex=$\mathbf{\theta}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\mathbf{\theta}$" title="$\mathbf{\theta}$" /></a> collects weight matrices and bias vectors
>* **Learning <a href="https://www.codecogs.com/eqnedit.php?latex=$\mathbf{\theta}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\mathbf{\theta}$" title="$\mathbf{\theta}$" /></a>:** approximate MLE (maximize variational lower bound w.r.t. <a href="https://www.codecogs.com/eqnedit.php?latex=$\mathbf{\theta}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\mathbf{\theta}$" title="$\mathbf{\theta}$" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=$\mathbf{\phi}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\mathbf{\phi}$" title="$\mathbf{\phi}$" /></a>)

><a href="https://www.codecogs.com/eqnedit.php?latex=$$\mathcal{L}_{\text{VAE}}&space;(\mathbf{\theta},\mathbf{\phi})&space;=&space;\sum^N_{n=1}&space;\mathbb{E}_{q_\mathbf{\phi}(\mathbf{z}^{(n)}|\mathbf{x}^{(n)})}&space;\left[&space;\log&space;\frac{p(\mathbf{x}^{(n)}|\mathbf{z}^{(n)},\mathbf{\theta})p(\mathbf{z}^{(n)})}&space;{q_\mathbf{\phi}&space;(\mathbf{z}^{(n)}|\mathbf{x}^{(n)})}&space;\right]$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\mathcal{L}_{\text{VAE}}&space;(\mathbf{\theta},\mathbf{\phi})&space;=&space;\sum^N_{n=1}&space;\mathbb{E}_{q_\mathbf{\phi}(\mathbf{z}^{(n)}|\mathbf{x}^{(n)})}&space;\left[&space;\log&space;\frac{p(\mathbf{x}^{(n)}|\mathbf{z}^{(n)},\mathbf{\theta})p(\mathbf{z}^{(n)})}&space;{q_\mathbf{\phi}&space;(\mathbf{z}^{(n)}|\mathbf{x}^{(n)})}&space;\right]$$" title="$$\mathcal{L}_{\text{VAE}} (\mathbf{\theta},\mathbf{\phi}) = \sum^N_{n=1} \mathbb{E}_{q_\mathbf{\phi}(\mathbf{z}^{(n)}|\mathbf{x}^{(n)})} \left[ \log \frac{p(\mathbf{x}^{(n)}|\mathbf{z}^{(n)},\mathbf{\theta})p(\mathbf{z}^{(n)})} {q_\mathbf{\phi} (\mathbf{z}^{(n)}|\mathbf{x}^{(n)})} \right]$$" /></a>

>* → **No parameter uncertainty estimates** (used to weight the information learned from old data)

* **Formulation** - VCL approach (continual learning)

>* Approximate full posterior over parametrs, <a href="https://www.codecogs.com/eqnedit.php?latex=$q_t(\mathbf{\theta})&space;\approx&space;p(\mathbf{\theta}|\mathcal{D}_{1:t})$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$q_t(\mathbf{\theta})&space;\approx&space;p(\mathbf{\theta}|\mathcal{D}_{1:t})$" title="$q_t(\mathbf{\theta}) \approx p(\mathbf{\theta}|\mathcal{D}_{1:t})$" /></a>
>* Maximize **full** variational lower bound w.r.t. <a href="https://www.codecogs.com/eqnedit.php?latex=$q_t$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$q_t$" title="$q_t$" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=$\phi$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\phi$" title="$\phi$" /></a>

><a href="https://www.codecogs.com/eqnedit.php?latex=$$\mathcal{L}^t_{\text{VAE}}&space;(q_t(\mathbf{\theta}),\mathbf{\phi})&space;=&space;\mathbb{E}_{q_t(\mathbf{\theta})}\left\{&space;\sum^{N_t}_{n=1}&space;\mathbb{E}_{q_\mathbf{\phi}(\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})}&space;\left[&space;\log&space;\frac{p(\mathbf{x}_t^{(n)}|\mathbf{z}_t^{(n)},\mathbf{\theta})p(\mathbf{z}_t^{(n)})}&space;{q_\mathbf{\phi}&space;(\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})}&space;\right]&space;\right\}&space;-\text{KL}(q_t(\mathbf{\theta})||q_{t-1}(\mathbf{\theta}))$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\mathcal{L}^t_{\text{VAE}}&space;(q_t(\mathbf{\theta}),\mathbf{\phi})&space;=&space;\mathbb{E}_{q_t(\mathbf{\theta})}\left\{&space;\sum^{N_t}_{n=1}&space;\mathbb{E}_{q_\mathbf{\phi}(\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})}&space;\left[&space;\log&space;\frac{p(\mathbf{x}_t^{(n)}|\mathbf{z}_t^{(n)},\mathbf{\theta})p(\mathbf{z}_t^{(n)})}&space;{q_\mathbf{\phi}&space;(\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})}&space;\right]&space;\right\}&space;-\text{KL}(q_t(\mathbf{\theta})||q_{t-1}(\mathbf{\theta}))$$" title="$$\mathcal{L}^t_{\text{VAE}} (q_t(\mathbf{\theta}),\mathbf{\phi}) = \mathbb{E}_{q_t(\mathbf{\theta})}\left\{ \sum^{N_t}_{n=1} \mathbb{E}_{q_\mathbf{\phi}(\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})} \left[ \log \frac{p(\mathbf{x}_t^{(n)}|\mathbf{z}_t^{(n)},\mathbf{\theta})p(\mathbf{z}_t^{(n)})} {q_\mathbf{\phi} (\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})} \right] \right\} -\text{KL}(q_t(\mathbf{\theta})||q_{t-1}(\mathbf{\theta}))$$" /></a>

>* $\boldsymbol{\phi}$: task-specific → likely to be beneficial to share (parts of) these encoder networks

* **Model Architecture**

>* Latent variables $\mathbf{z}$ → Intermediate-level representations $\mathbf{h}$ → Observations $\mathbf{x}$

>* **Architecture 1:** shared bottom network - suitable when data are composed of a common set of structural primitives (e.g. strokes)
>* **Architecture 2:** shared head network - information tend to be entirely encoded in bottom network

## 5. Related Work

* **Continual Learning for Deep Discriminative Models** (regularized MLE)

><a href="https://www.codecogs.com/eqnedit.php?latex=$$\mathcal{L}^t&space;(\boldsymbol{\theta})&space;=&space;\sum^{N_t}_{n=1}&space;\log&space;p(y_t^{(n)}&space;|&space;\boldsymbol{\theta},\mathbf{x}^{(n)}_t)&space;-&space;\frac{1}{2}&space;\lambda_t&space;(\boldsymbol{\theta}&space;-&space;\boldsymbol{\theta}_{t-1})^T&space;\Sigma^{-1}_{t-1}&space;(\boldsymbol{\theta}&space;-&space;\boldsymbol{\theta}_{t-1})$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\mathcal{L}^t&space;(\boldsymbol{\theta})&space;=&space;\sum^{N_t}_{n=1}&space;\log&space;p(y_t^{(n)}&space;|&space;\boldsymbol{\theta},\mathbf{x}^{(n)}_t)&space;-&space;\frac{1}{2}&space;\lambda_t&space;(\boldsymbol{\theta}&space;-&space;\boldsymbol{\theta}_{t-1})^T&space;\Sigma^{-1}_{t-1}&space;(\boldsymbol{\theta}&space;-&space;\boldsymbol{\theta}_{t-1})$$" title="$$\mathcal{L}^t (\boldsymbol{\theta}) = \sum^{N_t}_{n=1} \log p(y_t^{(n)} | \boldsymbol{\theta},\mathbf{x}^{(n)}_t) - \frac{1}{2} \lambda_t (\boldsymbol{\theta} - \boldsymbol{\theta}_{t-1})^T \Sigma^{-1}_{t-1} (\boldsymbol{\theta} - \boldsymbol{\theta}_{t-1})$$" /></a>

>* **ML Estimation** - set $\lambda_t = 0$

>* **MAP Estimation** - assume Gaussian prior <a href="https://www.codecogs.com/eqnedit.php?latex=$q(\mathbf{\theta}|\mathcal{D}_{1:t-1})=\mathcal{N}(\mathbf{\theta};\mathbf{\theta}_{t-1},\Sigma_{t-1}/\lambda_t)$&space;>&space;*&space;$\Sigma_t=?&space;\;&space;\rightarrow&space;\;&space;\Sigma_t=I$&space;and&space;use&space;CV&space;to&space;find&space;$\lambda_T$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$q(\mathbf{\theta}|\mathcal{D}_{1:t-1})=\mathcal{N}(\mathbf{\theta};\mathbf{\theta}_{t-1},\Sigma_{t-1}/\lambda_t)$&space;>&space;*&space;$\Sigma_t=?&space;\;&space;\rightarrow&space;\;&space;\Sigma_t=I$&space;and&space;use&space;CV&space;to&space;find&space;$\lambda_T$" title="$q(\mathbf{\theta}|\mathcal{D}_{1:t-1})=\mathcal{N}(\mathbf{\theta};\mathbf{\theta}_{t-1},\Sigma_{t-1}/\lambda_t)$ > * $\Sigma_t=? \; \rightarrow \; \Sigma_t=I$ and use CV to find $\lambda_T$" /></a> → catastrophic forgetting

>* **Laplace Propagation (LP)** ([Smola et al., 2004](https://papers.nips.cc/paper/2444-laplace-propagation.pdf)) - recursion for $\Sigma_t$ using Laplace's approximation
>  * Diagonal LP: retain only the diagonal terms of $\Sigma^{-1}_t$ to avoid computing full Hessian

><a href="https://www.codecogs.com/eqnedit.php?latex=$$\Sigma^{-1}_t&space;=&space;\Phi_t&space;&plus;&space;\Sigma^{-1}_{t-1}&space;\;\;\;,\;\;\;&space;\Phi_t&space;=&space;-&space;\nabla&space;\nabla_\mathbf{\theta}&space;\sum^{N_t}_{n=1}&space;\log&space;p(y^{(n)}_t&space;|&space;\mathbf{\theta},&space;\mathbf{x}^{(n)}_t)&space;\big|_{\mathbf{\theta}&space;=&space;\mathbf{\theta}_t}&space;\;\;\;,\;\;\;&space;\lambda_t&space;=&space;1$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\Sigma^{-1}_t&space;=&space;\Phi_t&space;&plus;&space;\Sigma^{-1}_{t-1}&space;\;\;\;,\;\;\;&space;\Phi_t&space;=&space;-&space;\nabla&space;\nabla_\mathbf{\theta}&space;\sum^{N_t}_{n=1}&space;\log&space;p(y^{(n)}_t&space;|&space;\mathbf{\theta},&space;\mathbf{x}^{(n)}_t)&space;\big|_{\mathbf{\theta}&space;=&space;\mathbf{\theta}_t}&space;\;\;\;,\;\;\;&space;\lambda_t&space;=&space;1$$" title="$$\Sigma^{-1}_t = \Phi_t + \Sigma^{-1}_{t-1} \;\;\;,\;\;\; \Phi_t = - \nabla \nabla_\mathbf{\theta} \sum^{N_t}_{n=1} \log p(y^{(n)}_t | \mathbf{\theta}, \mathbf{x}^{(n)}_t) \big|_{\mathbf{\theta} = \mathbf{\theta}_t} \;\;\;,\;\;\; \lambda_t = 1$$" /></a>

>* **Elastic Weight Consolidation (EWC)** ([Kirkpatrick et al., 2017](https://arxiv.org/pdf/1612.00796.pdf)) - modified diagonal LP
>  * Approximate the average Hessian of the likelihoods using Fisher information
><a href="https://www.codecogs.com/eqnedit.php?latex=$$\Phi_t&space;\approx&space;\text{diag}&space;\left(&space;\sum^{N_t}_{n=1}&space;\left(&space;\nabla_\mathbf{\theta}&space;\log&space;p(y^{(n)}_t|\mathbf{\theta},\mathbf{x}^{(n)}_t)&space;\right)^2&space;\;\Big|_{\mathbf{\theta}=\mathbf{\theta}_t}\right)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\Phi_t&space;\approx&space;\text{diag}&space;\left(&space;\sum^{N_t}_{n=1}&space;\left(&space;\nabla_\mathbf{\theta}&space;\log&space;p(y^{(n)}_t|\mathbf{\theta},\mathbf{x}^{(n)}_t)&space;\right)^2&space;\;\Big|_{\mathbf{\theta}=\mathbf{\theta}_t}\right)$$" title="$$\Phi_t \approx \text{diag} \left( \sum^{N_t}_{n=1} \left( \nabla_\mathbf{\theta} \log p(y^{(n)}_t|\mathbf{\theta},\mathbf{x}^{(n)}_t) \right)^2 \;\Big|_{\mathbf{\theta}=\mathbf{\theta}_t}\right)$$" /></a>
>  * Regularization term: introduce hyperparameter, remove prior, regularize intermediate estimates

><a href="https://www.codecogs.com/eqnedit.php?latex=$$\frac{1}{2}&space;(\boldsymbol{\theta}&space;-&space;\boldsymbol{\theta}_{t-1})^T&space;(\Sigma^{-1}_0&space;&plus;&space;\Sigma^{t-1}_{t'=1}&space;\Phi_{t'})&space;(\boldsymbol{\theta}&space;-&space;\boldsymbol{\theta}_{t-1})&space;\rightarrow&space;\frac{1}{2}&space;\sum^{t-1}_{t'=1}&space;\lambda_{t'}&space;(\boldsymbol{\theta}&space;-&space;\boldsymbol{\theta}_{t'-1})^T&space;\Phi_{t'}&space;(\boldsymbol{\theta}&space;-&space;\boldsymbol{\theta}_{t'-1})$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\frac{1}{2}&space;(\boldsymbol{\theta}&space;-&space;\boldsymbol{\theta}_{t-1})^T&space;(\Sigma^{-1}_0&space;&plus;&space;\Sigma^{t-1}_{t'=1}&space;\Phi_{t'})&space;(\boldsymbol{\theta}&space;-&space;\boldsymbol{\theta}_{t-1})&space;\rightarrow&space;\frac{1}{2}&space;\sum^{t-1}_{t'=1}&space;\lambda_{t'}&space;(\boldsymbol{\theta}&space;-&space;\boldsymbol{\theta}_{t'-1})^T&space;\Phi_{t'}&space;(\boldsymbol{\theta}&space;-&space;\boldsymbol{\theta}_{t'-1})$$" title="$$\frac{1}{2} (\boldsymbol{\theta} - \boldsymbol{\theta}_{t-1})^T (\Sigma^{-1}_0 + \Sigma^{t-1}_{t'=1} \Phi_{t'}) (\boldsymbol{\theta} - \boldsymbol{\theta}_{t-1}) \rightarrow \frac{1}{2} \sum^{t-1}_{t'=1} \lambda_{t'} (\boldsymbol{\theta} - \boldsymbol{\theta}_{t'-1})^T \Phi_{t'} (\boldsymbol{\theta} - \boldsymbol{\theta}_{t'-1})$$" /></a>

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

>* **Naïve approach:** apply VAE to $\mathcal{D}_t$ with parameters initialized at $\boldsymbol{\theta}_{t-1}$ → catastrophic forgetting
>* **Alternative:** add EWC regularization term to VAE objective & approximate marginal likelihood by variational lower bound
>  * Similar approximation can be used for **Hessian matrices for LP** and **$\Sigma^{-1}_t$ for SI** (Importance sampling: [Burda et al., 2016](https://arxiv.org/pdf/1509.00519.pdf))

><a href="https://www.codecogs.com/eqnedit.php?latex=$$\mathcal{L}^t_{EWC}&space;(\mathbf{\theta},\mathbf{\phi})&space;=&space;\sum^{N_t}_{n=1}&space;\mathbb{E}_{q_\mathbf{\phi}(\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})}&space;\left[&space;\log&space;\frac{p(\mathbf{x}_t^{(n)}|\mathbf{z}_t^{(n)},\boldsymbol{\theta})p(\mathbf{z}_t^{(n)})}&space;{q_\mathbf{\phi}&space;(\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})}&space;\right]&space;-&space;\frac{1}{2}&space;\sum^{t-1}_{t'=1}&space;\lambda_{t'}&space;(\mathbf{\theta}&space;-&space;\mathbf{\theta}_{t'-1})^T&space;\Phi_{t'}&space;(\mathbf{\theta}&space;-&space;\mathbf{\theta}_{t'-1})$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\mathcal{L}^t_{EWC}&space;(\mathbf{\theta},\mathbf{\phi})&space;=&space;\sum^{N_t}_{n=1}&space;\mathbb{E}_{q_\mathbf{\phi}(\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})}&space;\left[&space;\log&space;\frac{p(\mathbf{x}_t^{(n)}|\mathbf{z}_t^{(n)},\boldsymbol{\theta})p(\mathbf{z}_t^{(n)})}&space;{q_\mathbf{\phi}&space;(\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})}&space;\right]&space;-&space;\frac{1}{2}&space;\sum^{t-1}_{t'=1}&space;\lambda_{t'}&space;(\mathbf{\theta}&space;-&space;\mathbf{\theta}_{t'-1})^T&space;\Phi_{t'}&space;(\mathbf{\theta}&space;-&space;\mathbf{\theta}_{t'-1})$$" title="$$\mathcal{L}^t_{EWC} (\mathbf{\theta},\mathbf{\phi}) = \sum^{N_t}_{n=1} \mathbb{E}_{q_\mathbf{\phi}(\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})} \left[ \log \frac{p(\mathbf{x}_t^{(n)}|\mathbf{z}_t^{(n)},\boldsymbol{\theta})p(\mathbf{z}_t^{(n)})} {q_\mathbf{\phi} (\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})} \right] - \frac{1}{2} \sum^{t-1}_{t'=1} \lambda_{t'} (\mathbf{\theta} - \mathbf{\theta}_{t'-1})^T \Phi_{t'} (\mathbf{\theta} - \mathbf{\theta}_{t'-1})$$" /></a>

><a href="https://www.codecogs.com/eqnedit.php?latex=$$\Phi_t&space;\approx&space;\text{diag}&space;\left(&space;\sum^{N_t}_{n=1}&space;\left(&space;\nabla_\mathbf{\theta}&space;\mathbb{E}_{q_{\mathbf{\phi}}(\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})}&space;\left[&space;\log&space;\frac{p(\mathbf{x}_t^{(n)}|\mathbf{z}_t^{(n)},\mathbf{\theta})p(\mathbf{z}_t^{(n)})}&space;{q_\mathbf{\phi}&space;(\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})}&space;\right]&space;\right)^2&space;\;\Bigg|_{\mathbf{\theta}=\mathbf{\theta}_t}\right)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\Phi_t&space;\approx&space;\text{diag}&space;\left(&space;\sum^{N_t}_{n=1}&space;\left(&space;\nabla_\mathbf{\theta}&space;\mathbb{E}_{q_{\mathbf{\phi}}(\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})}&space;\left[&space;\log&space;\frac{p(\mathbf{x}_t^{(n)}|\mathbf{z}_t^{(n)},\mathbf{\theta})p(\mathbf{z}_t^{(n)})}&space;{q_\mathbf{\phi}&space;(\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})}&space;\right]&space;\right)^2&space;\;\Bigg|_{\mathbf{\theta}=\mathbf{\theta}_t}\right)$$" title="$$\Phi_t \approx \text{diag} \left( \sum^{N_t}_{n=1} \left( \nabla_\mathbf{\theta} \mathbb{E}_{q_{\mathbf{\phi}}(\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})} \left[ \log \frac{p(\mathbf{x}_t^{(n)}|\mathbf{z}_t^{(n)},\mathbf{\theta})p(\mathbf{z}_t^{(n)})} {q_\mathbf{\phi} (\mathbf{z}_t^{(n)}|\mathbf{x}_t^{(n)})} \right] \right)^2 \;\Bigg|_{\mathbf{\theta}=\mathbf{\theta}_t}\right)$$" /></a>
