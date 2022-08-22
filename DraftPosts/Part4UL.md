# Bayesian inference for a logistic regression model (Part 4)

## Part 4: Gradients and the Langevin algorithm

### Introduction

This is the fourth part in a series of posts on MCMC-based Bayesian inference for a [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) model. If you are new to this series, please go back to **[Part 1](https://darrenjw.wordpress.com/2022/08/07/bayesian-inference-for-a-logistic-regression-model-part-1/)**.

In the [previous post](https://darrenjw.wordpress.com/2022/08/14/bayesian-inference-for-a-logistic-regression-model-part-3/) we saw how the Metropolis algorithm could be used to generate a Markov chain targeting our posterior distribution. In high dimensions the diffusive nature of the Metropolis random walk proposal becomes increasingly inefficient. It is therefore natural to try and develop algorithms that use additional information about the target distribution. In the case of a differentiable log posterior target, a natural first step in this direction is to try and make use of gradient information.

## Gradient of a logistic regression model

There are various ways to derive the gradient of our logistic regression model, but it might be simplest to start from the first form of the log likelihood that we deduced in [Part 2](https://darrenjw.wordpress.com/2022/08/07/bayesian-inference-for-a-logistic-regression-model-part-2/):
$$
l(\beta;y) = y^\textsf{T}X\beta - \mathbf{1}^\textsf{T}\log(\mathbf{1}+\exp[X\beta])
$$
We can write this out in component form as
$$
l(\beta;y) = \sum_j\sum_j y_iX_{ij}\beta_j - \sum_i\log\left(1+\exp\left[\sum_jX_{ij}\beta_j\right]\right).
$$
Differentiating wrt $\beta_k$ gives
$$
\frac{\partial l}{\partial \beta_k} = \sum_i y_iX_{ik} - \sum_i \frac{\exp\left[\sum_j X_{ij}\beta_j\right]X_{ik}}{1+\exp\left[\sum_j X_{ij}\beta_j\right]}.
$$
It's then reasonably clear that stitching all of the partial derivatives together will give the gradient vector
$$
\nabla l = X^\textsf{T}\left[ y - \frac{\mathbf{1}}{\mathbf{1}+\exp[-X\beta]} \right].
$$
This is the gradient of the log likelihood, but we also need the gradient of the log prior. Since we are assuming independent $\beta_i \sim N(0,v_i)$ priors, it is easy to see that the gradient of the log prior is just $-\beta\circ v^{-1}$. It is the sum of these two terms that gives the gradient of the log posterior.

### R

In R we can implement our gradient function as
```R
glp = function(beta) {
    glpr = -beta/(pscale*pscale)
    gll = as.vector(t(X) %*% (y - 1/(1 + exp(-X %*% beta))))
    glpr + gll
}
```

### Python
In Python we could use
```python
def glp(beta):
    glpr = -beta/(pscale*pscale)
    gll = (X.T).dot(y - 1/(1 + np.exp(-X.dot(beta))))
    return (glpr + gll)
```
We don't really need a JAX version, since JAX can auto-diff the log posterior for us.

### Scala

```scala
  def glp(beta: DVD): DVD =
    val glpr = -beta /:/ pvar
    val gll = (X.t)*(y - ones/:/(ones + exp(-X*beta)))
    glpr + gll

```

### Haskell

Using [hmatrix](https://hackage.haskell.org/package/hmatrix) we could use something like
```haskell
glp :: Matrix Double -> Vector Double -> Vector Double -> Vector Double
glp x y b = let
  glpr = -b / (fromList [100.0, 1, 1, 1, 1, 1, 1, 1])
  gll = (tr x) #> (y - (scalar 1)/((scalar 1) + (cmap exp (-x #> b))))
  in glpr + gll
```
There's something interesting to say about Haskell and auto-diff, but getting into this now will be too much of a distraction. I may come back to it in some future post.

### Dex

Dex is differentiable, so we don't need a gradient function - we can just use `grad lpost`. However, for interest and comparison purposes we could nevertheless implement it directly with something like
```haskell
prscale = map (\ x. 1.0/x) pscale

def glp (b: (Fin 8)=>Float) : (Fin 8)=>Float =
  glpr = -b*prscale*prscale
  gll = (transpose x) **. (y - (map (\eta. 1.0/(1.0 + eta)) (exp (-x **. b))))
  glpr + gll
```

## Langevin diffusions

Now that we have a way of computing the gradient of the log of our target density we need some MCMC algorithms that can make good use of it. In this post we will look at a simple approximate MCMC algorithm derived from an overdamped [Langevin diffusion](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm) model. In subsequent posts we'll look at more sophisticated, exact MCMC algorithms.

The multivariate [stochastic differential equation](https://en.wikipedia.org/wiki/Stochastic_differential_equation) (SDE)
$$
dX_t = \frac{1}{2}\nabla\log\pi(X_t)dt + dW_t
$$
has $\pi(\cdot)$ as its equilibrium distribution. Informally, an SDE of this form is a continuous time process with infinitesimal transition kernel
$$
X_{t+dt}|(X_t=x_t) \sim N\left(x_t+\frac{1}{2}\nabla\log\pi(x_t)dt,\mathbf{I}dt\right).
$$
There are various more-or-less formal ways to see that $\pi(\cdot)$ is stationary. A good way is to check it satisfies the [Fokker–Planck equation](https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation) with zero LHS. A less formal approach would be to see that the infinitesimal transition kernel for the process satisfies detailed balance with $\pi(\cdot)$. 

Similar arguments show that for any fixed positive definite matrix $A$, the SDE
$$
dX_t = \frac{1}{2}A\nabla\log\pi(X_t)dt + A^{1/2}dW_t
$$
also has $\pi(\cdot)$ as a stationary distribution. It is quite common to choose a diagonal matrix $A$ to put the components of $X_t$ on a common scale.

### The unadjusted Langevin algorithm

Simulating exact sample paths from SDEs such as the overdamped Langevin diffusion model is typically difficult (though not necessarily impossible), so we instead want something simple and tractable as the basis of our MCMC algorithms. Here we will just simulate from the [Euler–Maruyama](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method) approximation of the process by choosing a small but finite time step $\Delta t$ and using the transition kernel
$$
X_{t+\Delta t}|(X_t=x_t) \sim N\left(x_t+\frac{1}{2}A\nabla\log\pi(x_t)\Delta t, A\Delta t\right)
$$
as the basis of our MCMC method. For sufficiently small $\Delta t$ this should accurately approximate the Langevin dynamics, leading to an equilibrium distribution very close to $\pi(\cdot)$. That said, we would like to choose $\Delta t$ as large as we can get away with, since that will lead to a more rapidly mixing MCMC chain. Below are some implementations of this kernel for a diagonal pre-conditioning matrix.

## Implementation

### R

We can create a kernel for the unadjusted Langevin algorithm in R with the following function.
```r
ulKernel = function(glpi, dt = 1e-4, pre = 1) {
    sdt = sqrt(dt)
    spre = sqrt(pre)
    advance = function(x) x + 0.5*pre*glpi(x)*dt
    function(x, ll) rnorm(p, advance(x), spre*sdt)
}
```
Here, we can pass in `pre`, which is expected to be a vector representing the diagonal of the pre-conditioning matrix, $A$. We can then use this kernel to generate an MCMC chain as we have seen previously. See the [full runnable script](https://github.com/darrenjw/logreg/blob/main/R/fit-ul.R) for further details.

### Python

```python
def ulKernel(glpi, dt = 1e-4, pre = 1):
    p = len(init)
    sdt = np.sqrt(dt)
    spre = np.sqrt(pre)
    advance = lambda x: x + 0.5*pre*glpi(x)*dt
    def kernel(x):
        return advance(x) + np.random.randn(p)*spre*sdt
    return kernel
```
See the [full runnable script](https://github.com/darrenjw/logreg/blob/main/Python/fit-np-ul.py) for further details.

#### JAX

```python
def ulKernel(lpi, dt = 1e-4, pre = 1):
    p = len(init)
    glpi = jit(grad(lpi))
    sdt = jnp.sqrt(dt)
    spre = jnp.sqrt(pre)
    advance = jit(lambda x: x + 0.5*pre*glpi(x)*dt)
    @jit
    def kernel(key, x):
        return advance(x) + jax.random.normal(key, [p])*spre*sdt
    return kernel
```
Note how for JAX we can just pass in the log posterior, and the gradient function can be obtained by automatic differentiation. See the [full runnable script](https://github.com/darrenjw/logreg/blob/main/Python/fit-jax-ul.py) for further details.

### Scala

```scala
def ulKernel(glp: DVD => DVD, pre: DVD, dt: Double): DVD => DVD =
  val sdt = math.sqrt(dt)
  val spre = sqrt(pre)
  def advance(beta: DVD): DVD =
    beta + (0.5*dt)*(pre*:*glp(beta))
  beta => advance(beta) + sdt*spre.map(Gaussian(0,_).sample())
```
See the [full runnable script](https://github.com/darrenjw/logreg/blob/main/Scala/lr/src/main/scala/fit-ul.scala) for further details.

### Haskell

```haskell
ulKernel :: (StatefulGen g m) =>
  (Vector Double -> Vector Double) -> Vector Double -> Double -> g ->
  Vector Double -> m (Vector Double)
ulKernel glpi pre dt g beta = do
  let sdt = sqrt dt
  let spre = cmap sqrt pre
  let p = size pre
  let advance beta = beta + (scalar (0.5*dt))*pre*(glpi beta)
  zl <- (replicateM p . genContVar (normalDistr 0.0 1.0)) g
  let z = fromList zl
  return $ advance(beta) + (scalar sdt)*spre*z
```
See the [full runnable script](https://github.com/darrenjw/logreg/blob/main/Haskell/lr/app/Lang.hs) for further details.

### Dex

In Dex we can write a function that accepts a gradient function
```haskell
def ulKernel {n} (glpi: (Fin n)=>Float -> (Fin n)=>Float)
    (pre: (Fin n)=>Float) (dt: Float)
    (b: (Fin n)=>Float) (k: Key) : (Fin n)=>Float =
  sdt = sqrt dt
  spre = sqrt pre
  b + (((0.5)*dt) .* (pre*(glpi b))) +
    (sdt .* (spre*(randn_vec k)))
```
or we can write a function that accepts a log posterior, and uses auto-diff to construct the gradient
```haskell
def ulKernel {n} (lpi: (Fin n)=>Float -> Float)
    (pre: (Fin n)=>Float) (dt: Float)
    (b: (Fin n)=>Float) (k: Key) : (Fin n)=>Float =
  glpi = grad lpi
  sdt = sqrt dt
  spre = sqrt pre
  b + ((0.5)*dt) .* (pre*(glpi b)) +
    sdt .* (spre*(randn_vec k))
```
and since Dex is statically typed, we can't easily mix these functions up.

See the full runnable scripts, [without](https://github.com/darrenjw/logreg/blob/main/Dex/fit-ul.dx) and [with](https://github.com/darrenjw/logreg/blob/main/Dex/fit-ul-ad.dx) auto-diff.

## Next steps

In this post we have seen how to construct an MCMC algorithm that makes use of gradient information. But this algorithm is approximate. In the next post we'll see how to correct for the approximation by using the Langevin updates as proposals within a Metropolis-Hastings algorithm.


