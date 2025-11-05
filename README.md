# GMM-From-Scratch-EM-Algorithm
"ML models implemented from scratch using NumPy and Pandas only"

🧠 Gaussian Mixture Model (GMM) from Scratch

with EM Algorithm, BIC/AIC, and Silhouette Score


---

📘 Overview

This project implements the Gaussian Mixture Model (GMM) completely from scratch using Python and NumPy, without scikit-learn’s GMM class.
It also includes model selection using:

BIC (Bayesian Information Criterion)

AIC (Akaike Information Criterion)

Silhouette Score


The goal is to automatically find the optimal number of clusters (K) and visualize results with Gaussian ellipses.


---

🧩 Algorithm Steps

1️⃣ Initialization

Randomly initialize:

 $$\mu_k$$→ Mean of each cluster

 $$\Sigma_k$$→ Covariance matrix of each cluster

$$\pi_k$$ → Mixing coefficients


$$\pi_k = \frac{1}{K}$$


---

2️⃣ Expectation Step (E-Step)

Compute the responsibility of cluster  for data point :

$$\gamma_{ik} = \frac{\pi_k \, \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_i | \mu_j, \Sigma_j)}$$

Where  is the multivariate Gaussian probability density:

$$\mathcal{N}(x_i | \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2}(x_i - \mu_k)^T \Sigma_k^{-1}(x_i - \mu_k)\right)$$


---

3️⃣ Maximization Step (M-Step)

Update parameters based on responsibilities:

$$N_k = \sum_{i=1}^{n} \gamma_{ik}$$

$$\mu_k$$ = $$\frac{1}{N_k} \sum_{i=1}^{n} \gamma_{ik} x_i $$

$$\Sigma_k$$ = $$\frac{1}{N_k} \sum_{i=1}^{n} \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T$$

$$\pi_k$$ = $$\frac{N_k}{n} $$

Repeat E-Step and M-Step until log-likelihood converges.


---

4️⃣ Log-Likelihood

$$\mathcal{L}$$ = $$\sum_{i=1}^{n} \log\left(\sum_{k=1}^{K} \pi_k \$$, $$\mathcal{N}(x_i | \mu_k, \Sigma_k)\right)$$


---

📊 Model Selection Metrics

AIC (Akaike Information Criterion)

AIC = -2 $$\ln(\hat{L})$$ + 2p

BIC (Bayesian Information Criterion)

BIC = -2 $$\ln(\hat{L^})$$ + p $$\ln(n)$$

L_^: maximum likelihood

p: number of parameters

n: number of data points


Silhouette Score

s = $$\frac{b - a}{\max(a, b)}$$

a: average intra-cluster distance

b: average nearest-cluster distance



---

⚙️ Implementation Outline
###
for K in range(2, 8):
    means, covs, weights = initialize_params(X, K)
    for iteration in range(100):
        resp = e_step(X, means, covs, weights, K)
        means, covs, weights = m_step(X, resp, K)
        log_likelihood = compute_log_likelihood(X, means, covs, weights, K)
        if convergence: break
    bic, aic, sil = compute_metrics(log_likelihood, K, X, resp)
###

---

🎯 Automatic Model Selection

The code automatically:

Calculates BIC, AIC, and Silhouette for each K

Selects the best K (based on Silhouette by default)

Visualizes metric trends vs K



---

🧾 Visualization

1️⃣ Model Selection Plot

Shows how BIC, AIC, and Silhouette vary with different cluster counts.

📈 Lower BIC/AIC → Better model fit  
📈 Higher Silhouette → Better separation

2️⃣ Cluster Visualization

Each cluster is visualized with a Gaussian ellipse representing its covariance.


---

### 📁 Project Structure

📦 GMM_from_Scratch/
├── GMM_scratch.ipynb        # Full implementation
├── README.md                # This file
├── data/                    # Dataset (optional)
└── images/                  # Plots and results


---

🧠 Key Insights

Concept	                          Meaning

E-Step	                           Estimate probability of each point belonging to each cluster
M-Step	                           Recalculate parameters using these probabilities
Log-Likelihood	                   Measures how well the model fits the data
BIC/AIC	                           Penalize overfitting with too many clusters
Silhouette Score	                 Validates cluster separation and cohesion



---

🧮 Example Output

Metric	Best K

BIC	5
AIC	7
Silhouette	5


✅ Final model automatically selects K = 5 and visualizes Gaussian ellipses.


---

🚀 Libraries Used
###
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
###

---

💡 Applications

Customer segmentation

Anomaly detection

Speaker recognition

Image compression

Soft clustering of user behaviors



---

🧭 Next Steps

1. Apply the model on real datasets (e.g., Mall_Customers.csv)


2. Compare results with K-Means and Hierarchical Clustering

   🔢 Mathematical Appendix — Derivation of EM updates for GMM

Setup.
Data , . Mixture of  Gaussians parameterized by  with .
Mixture density:

$$p(x_i\mid\Theta)$$=$$\sum_{k=1}^K \pi_k\$$,$$\mathcal{N}(x_i\mid\mu_k,\Sigma_k)$$.

We maximize log-likelihood .
Because of the log-of-sum, use EM with latent one-hot  indicating component.


---

1. E-step — responsibilities (posterior probabilities)

Define the posterior responsibility .
By Bayes rule:

$$\gamma_{ik}$$
$$\y_ik$$= $$\frac{\pi_k^{(t)}\,\mathcal{N}(x_i\mid\mu_k^{(t)},\Sigma_k^{(t)})}
{\sum_{j=1}^K \pi_j^{(t)}\,\mathcal{N}(x_i\mid\mu_j^{(t)}$$,$$\Sigma_j^{(t)})}$$.

Interpretation: soft assignment of  to component . These  are computed using the current parameters.


---

2. M-step — maximize expected complete-data log-likelihood

Define the expected complete-data log-likelihood (the Q-function):

$$Q(\Theta \mid \Theta^{(t)})$$ = $$\mathbb{E}_{Z\mid X$$,$$\Theta^{(t)}}[\log p(X,Z\mid\Theta)]$$
= $$\sum_{i=1}^n\sum_{k=1}^K \gamma_{ik}\$$,$$\log\big(\pi_k\$$,$$\mathcal{N}(x_i\mid\mu_k$$,$$\Sigma_k)\big)$$.

Q=$$\sum_{k=1}^K \sum_{i=1}^n \gamma_{ik}\big(\log\pi_k$$ + $$\log\mathcal{N}(x_i\mid\mu_k$$,$$\Sigma_k)\big)$$.

We maximize Q w.r.t. $$\pi_k$$, $$\mu_k$$, $$\Sigma_k$$ subject to .


---

(a) Update for $$\mu_k$$

Keep  fixed and maximize w.r.t. . Use the Gaussian log-density:

$$\log\mathcal{N}(x_i\mid\mu_k,\Sigma_k)$$
= $$-\frac{d}{2}\log(2\pi)-\frac{1}{2}\log|\Sigma_k| -\frac{1}{2}(x_i-\mu_k)^\top\Sigma_k^{-1}(x_i-\mu_k)$$.

$$Q_{\mu_k}$$ = -$$\frac{1}{2}\sum_{i=1}^n \gamma_{ik}(x_i-\mu_k)^\top\Sigma_k^{-1}(x_i-\mu_k)$$.

$$\frac{\partial Q_{\mu_k}}{\partial\mu_k}$$
= -$$\frac{1}{2}\sum_{i=1}^n \gamma_{ik}\big(-2\Sigma_k^{-1}(x_i-\mu_k)\big)$$
= $$\Sigma_k^{-1}\sum_{i=1}^n \gamma_{ik}(x_i-\mu_k)$$=0.

$$\sum_{i=1}^n \gamma_{ik}(x_i-\mu_k)$$=0
$$\quad\Rightarrow\quad$$
$$\mu_k$$ = $$\frac{\sum_{i=1}^n \gamma_{ik} x_i}{\sum_{i=1}^n \gamma_{ik}}$$.

$$\boxed{\mu_k$$ = $$\frac{1}{N_k}\sum_{i=1}^n \gamma_{ik} x_i.}$$


---

(b) Update for $$\Sigma_k$$

Maximize  Q w.r.t. $$\sigma_k$$. Collect terms depending on :

$$Q_{\Sigma_k}$$ = -$$\frac{1}{2}\sum_{i=1}^n \gamma_{ik}\Big(\log|\Sigma_k| + (x_i-\mu_k)^\top\Sigma_k^{-1}(x_i-\mu_k)\Big)$$.

$$\frac{\partial Q_{\Sigma_k}}{\partial \Sigma_k^{-1}}$$ = $$\frac{1}{2}\sum_{i}\gamma_{ik}\big((x_i-\mu_k)(x_i-\mu_k)^\top - \Sigma_k\big)$$=0.

$$\sum_{i}\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top$$ = $$N_k \Sigma_k$$.

$$\boxed{\Sigma_k = \frac{1}{N_k}\sum_{i=1}^n \gamma_{ik} (x_i-\mu_k)(x_i-\mu_k)^\top.}$$

(Practically add small regularizer:  to avoid singularity.)


---

(c) Update for  (mixing coefficients) via Lagrange multiplier

Maximize

$$Q_{\pi}$$ = $$\sum_{i=1}^n\sum_{k=1}^K \gamma_{ik}\log\pi_k$$

$$\mathcal{L}$$ = $$\sum_{k}\Big(\sum_{i}\gamma_{ik}\Big)\log\pi_k + \lambda\Big(1-\sum_{k}\pi_k\Big)$$.

$$\frac{\partial\mathcal{L}}{\partial\pi_k}$$ = $$\frac{N_k}{\pi_k}$$ - $$\lambda$$ = 0
$$\quad\Rightarrow\quad$$
$$\pi_k$$ = $$\frac{N_k}{\lambda}$$.

$$\sum_{k}\pi_k$$ = $$\frac{1}{\lambda}\sum_k N_k$$ = $4\frac{n}{\lambda}$$ = 1
$$\quad\Rightarrow\quad \lambda$$ = n.

$$\boxed{\pi_k$$ = $$\frac{N_k}{n}$$ = $$\frac{1}{n}\sum_{i=1}^n \gamma_{ik}.}$$


---

3. Log-likelihood (monitor convergence)

At each iteration compute log-likelihood:

$$\mathcal{L}(\Theta)$$ = $$\sum_{i=1}^n \log\Big(\sum_{k=1}^K \pi_k\,\mathcal{N}(x_i\mid\mu_k,\Sigma_k)\Big)$$.


---

✅ Final EM update summary (boxed)

Let  and  from E-step.

$$\boxed{\gamma_{ik}$$ = $$\frac{\pi_k\,\mathcal{N}(x_i\mid\mu_k,\Sigma_k)}{\sum_{j=1}^K \pi_j\,\mathcal{N}(x_i\mid\mu_j,\Sigma_j)}}\quad\text{(E-step)}$$

$$\boxed{\mu_k $$= $$\frac{1}{N_k}\sum_{i=1}^n \gamma_{ik} x_i}$$
$$\quad,\quad$$
$$\boxed{\Sigma_k = \frac{1}{N_k}\sum_{i=1}^n \gamma_{ik} (x_i-\mu_k)(x_i-\mu_k)^\top}$$
$$\quad,\quad$$
$$\boxed{\pi_k$$ = $$\frac{N_k}{n}}$$
$$\quad\text{(M-step)}$$


---

⚙️ Practical notes (important for implementation)

Regularization: add  to  each M-step to prevent singular matrices (common when a component collapses).

Initialization: KMeans or k-means++ centroids help convergence and avoid bad local optima.

Numerical stability: compute Gaussian pdfs in log-space if  large to avoid underflow; use log-sum-exp trick for responsibilities and likelihood.

Convergence criterion: use change in log-likelihood or relative parameter change.

Model selection: use BIC/AIC and silhouette to choose .


4. Extend to Variational Bayesian GMM (VBGMM) for adaptive cluster learning
