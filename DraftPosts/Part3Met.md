# Bayesian inference for a logistic regression model (Part 3)

## Part 3: The Metropolis algorithm

### Introduction

This is the third part in a series of posts on MCMC-based Bayesian inference for a [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) model. If you are new to this series, please go back to **[Part 1](https://darrenjw.wordpress.com/2022/08/07/bayesian-inference-for-a-logistic-regression-model-part-1/)**.

In the [previous post](https://darrenjw.wordpress.com/2022/08/07/bayesian-inference-for-a-logistic-regression-model-part-2/) we derived the log posterior for the model and implemented it in a variety of programming languages and libraries. In this post we will construct a Markov chain having the posterior as its equilibrium.

## MCMC

### Detailed balance

A homogeneous [Markov chain](https://en.wikipedia.org/wiki/Markov_chain) with transition kernel $p(\theta_{n+1}|\theta_n)$ is said to satisfy [detailed balance](https://en.wikipedia.org/wiki/Detailed_balance) for some target distribution $\pi(\theta)$ if
$$
\pi(\theta)p(\theta'|\theta) = \pi(\theta')p(\theta|\theta'),
\quad \forall \theta, \theta'
$$
Integrating both sides wrt $\theta$ gives
$$
\int_\Theta \pi(\theta)p(\theta'|\theta)d\theta = \pi(\theta'),
$$
from which it is clear that $\pi(\cdot)$ is a *stationary* distribution of the chain (and the chain is reversible). Under fairly mild regularity conditions we expect $\pi(\cdot)$ to be the equilibrium distribution of the chain.

For a given target $\pi(\cdot)$ we would like to find an easy-to-sample-from transition kernel $p(\cdot|\cdot)$ that satisfies detailed balance. This will then give us a way to (asymptotically) generate samples from our target.

In the context of Bayesian inference, the target $\pi(\theta)$ will typically be the *posterior distribution*, which in the previous post we wrote as $\pi(\theta|y)$. Here we drop the notational dependence on $y$, since MCMC can be used for any target distribution of interest.

### Metropolis-Hastings

Suppose we have a fairly arbitrary easy-to-sample-from transition kernel $q(\theta_{n+1}|\theta_n)$ and a target of interest, $\pi(\cdot)$. [Metropolis-Hastings](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) (M-H) is a strategy for using $q(\cdot|\cdot)$ to construct a new transition kernel $p(\cdot|\cdot)$ satisfying detailed balance for $\pi(\cdot)$.

The kernel $p(\theta'|\theta)$ can be described algorithmically as follows:
1. Call the current state of the chain $\theta$. Generate a *proposal* $\theta^\star$ by simulating from $q(\theta^\star|\theta)$.
2. Compute the *acceptance probability*
$$
\alpha(\theta^\star|\theta) = \min\left[1,\frac{\pi(\theta^\star)q(\theta|\theta^\star)}{\pi(\theta)q(\theta^\star|\theta)}\right].
$$
3. With probability $\alpha(\theta^\star|\theta)$ return new state $\theta'=\theta^\star$, otherwise return $\theta'=\theta$.

It is clear from the algorithmic description that this kernel will have a point mass at $\theta'=\theta$, but that for $\theta'\not=\theta$ the transition kernel will be $p(\theta'|\theta)=q(\theta'|\theta)\alpha(\theta'|\theta)$. But then
$$
\pi(\theta)p(\theta'|\theta) = \min[\pi(\theta)q(\theta'|\theta),\pi(\theta')q(\theta|\theta')]
$$
is symmetric in $\theta$ and $\theta'$, and so detailed balance is satisfied. Since detailed balance is trivial at the point mass at $\theta=\theta'$ we are done.

#### Metropolis algorithm

It is often convenient to generate proposals perturbatively, using a distribution that is symmetric about the current state of the chain. But then $q(\theta'|\theta)=q(\theta|\theta')$, and so $q(\cdot|\cdot)$ drops out of the acceptance probability. This is the Metropolis algorithm.

#### Some computational tricks

To generate an event with probability $\alpha(\theta^\star|\theta)$, we can generate a $u\sim U(0,1)$ and accept if $u < \alpha(\theta^\star|\theta)$. This is convenient for several reasons. First, it means that we can ignore the "min", and just accept if
$$
u < \frac{\pi(\theta^\star)q(\theta|\theta^\star)}{\pi(\theta)q(\theta^\star|\theta)}
$$
since $u\leq 1$ regardless. Better still, we can take logs, and accept if
$$
\log u < \log\pi(\theta^\star) - \log\pi(\theta) + \log q(\theta|\theta^\star) - \log q(\theta^\star|\theta),
$$
so there is no need to evaluate any raw densities. Again, in the case of a symmetric proposal distribution, the $q(\cdot|\cdot)$ terms can be dropped.

Another trick worth noting is that in the case of the simple M-H algorithm described, *using a single update for the entire state space* (and *not* multiple component-wise updates, for example), and assuming that the same M-H kernel is used repeatedly to generate successive states of a Markov chain, then the $\log \pi(\theta)$ term (which in the context of Bayesian inference will typically be the log posterior) will have been computed at the previous update (irrespective of whether or not the previous move was accepted). So if we are careful about how we pass on that old value to the next iteration, we can avoid recomputing the log posterior, and our algorithm will only require one log posterior evaluation per iteration rather than two. In functional programming languages it is often convenient to pass around this current log posterior density evaluation explicitly, effectively augmenting the state space of the Markov chain to include the log posterior density.

## HoF for a M-H kernel

Since I'm a fan of functional programming, we will adopt a functional style throughout, and start by creating a [higher-order function](https://en.wikipedia.org/wiki/Higher-order_function) (HoF) that accepts a log-posterior and proposal kernel as input and returns a Metropolis kernel as output.

### R

In R we can write a function to create a M-H kernel as follows.
```R
mhKernel = function(logPost, rprop, dprop = function(new, old, ...) { 1 })
    function(x, ll) {
        prop = rprop(x)
        llprop = logPost(prop)
        a = llprop - ll + dprop(x, prop) - dprop(prop, x)
        if (log(runif(1)) < a)
            list(x=prop, ll=llprop)
        else
            list(x=x, ll=ll)
    }
```
Note that the kernel returned requires as input both a current state `x` and its associated log-posterior, `ll`. The new state and log-posterior densities are returned.

We need to use this transition kernel to simulate a Markov chain by successive substitution of newly simulated values back into the kernel. In more sophisticated programming languages we will use [streams](https://en.wikipedia.org/wiki/Stream_(computer_science)) for this, but in R we can just use a `for` loop to sample values and write the states into the rows of a matrix.

```R
mcmc = function(init, kernel, iters = 10000, thin = 10, verb = TRUE) {
    p = length(init)
    ll = -Inf
    mat = matrix(0, nrow = iters, ncol = p)
    colnames(mat) = names(init)
    x = init
    if (verb) 
        message(paste(iters, "iterations"))
    for (i in 1:iters) {
        if (verb) 
            message(paste(i, ""), appendLF = FALSE)
        for (j in 1:thin) {
            pair = kernel(x, ll)
            x = pair$x
            ll = pair$ll
            }
        mat[i, ] = x
        }
    if (verb) 
        message("Done.")
    mat
}
```
Then, in the context of our running logistic regression example, and using the log-posterior from the previous post, we can construct our kernel and run it as follows.
```R
pre = c(10.0,1,1,1,1,1,5,1)
out = mcmc(init, mhKernel(lpost,
          function(x) x + pre*rnorm(p, 0, 0.02)), thin=1000)
```
Note the use of a symmetric proposal, so the proposal density is not required. Also note the use of a larger proposal variance for the intercept term and the second last covariate. See the [full runnable script](https://github.com/darrenjw/logreg/blob/main/R/fit-bayes.R) for further details.

### Python

We can do something very similar to R in Python using NumPy. Our HoF for constructing a M-H kernel is
```python
def mhKernel(lpost, rprop, dprop = lambda new, old: 1.):
    def kernel(x, ll):
        prop = rprop(x)
        lp = lpost(prop)
        a = lp - ll + dprop(x, prop) - dprop(prop, x)
        if (np.log(np.random.rand()) < a):
            x = prop
            ll = lp
        return x, ll
    return kernel
```
Our Markov chain runner function is
```python
def mcmc(init, kernel, thin = 10, iters = 10000, verb = True):
    p = len(init)
    ll = -np.inf
    mat = np.zeros((iters, p))
    x = init
    if (verb):
        print(str(iters) + " iterations")
    for i in range(iters):
        if (verb):
            print(str(i), end=" ", flush=True)
        for j in range(thin):
            x, ll = kernel(x, ll)
        mat[i,:] = x
    if (verb):
        print("\nDone.", flush=True)
    return mat
```
We can use this code in the context of our logistic regression example as follows.
```python
pre = np.array([10.,1.,1.,1.,1.,1.,5.,1.])

def rprop(beta):
    return beta + 0.02*pre*np.random.randn(p)

out = mcmc(init, mhKernel(lpost, rprop), thin=1000)
```
See the [full runnable script](https://github.com/darrenjw/logreg/blob/main/Python/fit-numpy.py) for further details.

#### JAX

The above R and Python scripts are fine, but both languages are rather slow for this kind of workload. Fortunately it's rather straightforward to convert the Python code to JAX to obtain quite amazing speed-up. We can write our M-H kernel as
```python
def mhKernel(lpost, rprop, dprop = jit(lambda new, old: 1.)):
    @jit
    def kernel(key, x, ll):
        key0, key1 = jax.random.split(key)
        prop = rprop(key0, x)
        lp = lpost(prop)
        a = lp - ll + dprop(x, prop) - dprop(prop, x)
        accept = (jnp.log(jax.random.uniform(key1)) < a)
        return jnp.where(accept, prop, x), jnp.where(accept, lp, ll)
    return kernel
```
and our MCMC runner function as
```python
def mcmc(init, kernel, thin = 10, iters = 10000):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, iters)
    @jit
    def step(s, k):
        [x, ll] = s
        x, ll = kernel(k, x, ll)
        s = [x, ll]
        return s, s
    @jit
    def iter(s, k):
        keys = jax.random.split(k, thin)
        _, states = jax.lax.scan(step, s, keys)
        final = [states[0][thin-1], states[1][thin-1]]
        return final, final
    ll = -np.inf
    x = init
    _, states = jax.lax.scan(iter, [x, ll], keys)
    return states[0]
```
There are really only two slightly tricky things about this code.

The first relates to the way JAX handles pseudo-random numbers. Since JAX is a *pure functional* eDSL, it can't be used in conjunction with the typical pseudo-random number generators often used in *imperative* programming languages which rely on a *global mutable state*. This can be dealt with reasonably straightforwardly by explicitly passing around the random number state. There is a standard way of doing this that has been common practice in functional programming languages for decades. However, this standard approach is very sequential, and so doesn't work so well in a parallel context. JAX therefore uses a *splittable* random number generator, where new states are created by *splitting* the current state into two (or more). We'll come back to this when we get to the Haskell examples.

The second thing that might be unfamiliar to imperative programmers is the use of the *scan* operation (`jax.lax.scan`) to generate the Markov chain rather than a "for" loop. But *scans* are standard operations in most functional programming languages. 

We can then call this code for our logistic regression example with
```python
pre = jnp.array([10.,1.,1.,1.,1.,1.,5.,1.]).astype(jnp.float32)

@jit
def rprop(key, beta):
    return beta + 0.02*pre*jax.random.normal(key, [p])

out = mcmc(init, mhKernel(lpost, rprop), thin=1000)
```
See the [full runnable script](https://github.com/darrenjw/logreg/blob/main/Python/fit-jax2.py) for further details.

### Scala

In Scala we can use a similar approach to that already seen for defining a HoF to return a M-H kernel.
```scala
def mhKernel[S](
    logPost: S => Double, rprop: S => S,
    dprop: (S, S) => Double = (n: S, o: S) => 1.0
  ): ((S, Double)) => (S, Double) =
    val r = Uniform(0.0,1.0)
    state =>
      val (x0, ll0) = state
      val x = rprop(x0)
      val ll = logPost(x)
      val a = ll - ll0 + dprop(x0, x) - dprop(x, x0)
      if (math.log(r.draw()) < a)
        (x, ll)
      else
        (x0, ll0)
```
Note that Scala's static typing does not prevent us from defining a function that is *polymorphic* in the type of the chain state, which we here call `S`. Also note that we are adopting a pragmatic approach to random number generation, exploiting the fact that Scala is not a *pure* functional language, using a mutable generator, and omitting to capture the non-determinism of the `rprop` function (and the returned kernel) in its type signature. In Scala this is a choice, and we could adopt a purer approach if preferred. We'll see what such an approach will look like in Haskell, coming up next.

Now that we have the kernel, we don't need to write an explicit runner function since Scala has good support for streaming data. There are many more-or-less sophisticated ways that we can work with data streams in Scala, and the choice depends partly on how pure one is being about tracking effects (such as non-determinism), but here I'll just use the simple `LazyList` from the standard library for unfolding the kernel into an *infinite* MCMC chain before thinning and truncating appropriately.
```scala
  val pre = DenseVector(10.0,1.0,1.0,1.0,1.0,1.0,5.0,1.0)
  def rprop(beta: DVD): DVD = beta + pre *:* (DenseVector(Gaussian(0.0,0.02).sample(p).toArray))
  val kern = mhKernel(lpost, rprop)
  val s = LazyList.iterate((init, -Inf))(kern) map (_._1)
  val out = s.drop(150).thin(1000).take(10000)
```
See the [full runnable script](https://github.com/darrenjw/logreg/blob/main/Scala/lr/src/main/scala/fit-bayes.scala) for further details.

### Haskell

Since Haskell is a pure functional language, we need to have some convention regarding pseudo-random number generation. Haskell supports several styles. The most commonly adopted approach wraps a mutable generator up in a monad. The typical alternative is to use a pure functional generator and either explicitly thread the state through code or hide this in a monad similar to the standard approach. However, Haskell also supports the use of *splittable* generators, so we can consider all three approaches for comparative purposes. The approach taken does affect the code and the type signatures, and even the streaming data abstractions most appropriate for chain generation.

Starting with a HoF for producing a Metropolis kernel, an approach using the standard monadic generators could like like
```haskell
mKernel :: (StatefulGen g m) => (s -> Double) -> (s -> g -> m s) -> 
           g -> (s, Double) -> m (s, Double)
mKernel logPost rprop g (x0, ll0) = do
  x <- rprop x0 g
  let ll = logPost(x)
  let a = ll - ll0
  u <- (genContVar (uniformDistr 0.0 1.0)) g
  let next = if ((log u) < a)
        then (x, ll)
        else (x0, ll0)
  return next
```
Note how non-determinism is captured in the type signatures by the monad `m`. The explicit pure approach is to thread the generator through non-deterministic functions.
```haskell
mKernelP :: (RandomGen g) => (s -> Double) -> (s -> g -> (s, g)) -> 
            g -> (s, Double) -> ((s, Double), g)
mKernelP logPost rprop g (x0, ll0) = let
  (x, g1) = rprop x0 g
  ll = logPost(x)
  a = ll - ll0
  (u, g2) = uniformR (0, 1) g1
  next = if ((log u) < a)
        then (x, ll)
        else (x0, ll0)
  in (next, g2)
```
Here the updated random number generator state is returned from each non-deterministic function for passing on to subsequent non-deterministic functions. This explicit sequencing of operations makes it possible to wrap the generator state in a state monad giving code very similar to the stateful monadic generator approach, but as already discussed, the sequential nature of this approach makes it unattractive in parallel and concurrent settings.

Fortunately the standard Haskell pure generator is *splittable*, meaning that we can adopt a splitting approach similar to JAX if we prefer, since this is much more parallel-friendly.
```haskell
mKernelP :: (RandomGen g) => (s -> Double) -> (s -> g -> s) -> 
            g -> (s, Double) -> (s, Double)
mKernelP logPost rprop g (x0, ll0) = let
  (g1, g2) = split g
  x = rprop x0 g1
  ll = logPost(x)
  a = ll - ll0
  u = unif g2
  next = if ((log u) < a)
        then (x, ll)
        else (x0, ll0)
  in next
```
Here non-determinism is signalled by passing a generator state (often called a "key" in the context of splittable generators) into a function. Functions receiving a key are responsible for *splitting* it to ensure that no key is ever used more than once.

Once we have a kernel, we need to unfold our Markov chain. When using the monadic generator approach, it is most natural to unfold using a *monadic stream*
```haskell
mcmc :: (StatefulGen g m) =>
  Int -> Int -> s -> (g -> s -> m s) -> g -> MS.Stream m s
mcmc it th x0 kern g = MS.iterateNM it (stepN th (kern g)) x0

stepN :: (Monad m) => Int -> (a -> m a) -> (a -> m a)
stepN n fa = if (n == 1)
  then fa
  else (\x -> (fa x) >>= (stepN (n-1) fa))
```
whereas for the explicit approaches it is more natural to unfold into a regular infinite data stream. So, for the explicit sequential approach we could use
```haskell
mcmcP :: (RandomGen g) => s -> (g -> s -> (s, g)) -> g -> DS.Stream s
mcmcP x0 kern g = DS.unfold stepUf (x0, g)
  where
    stepUf xg = let
      (x1, g1) = kern (snd xg) (fst xg)
      in (x1, (x1, g1))
```
and with the splittable approach we could use
```haskell
mcmcP :: (RandomGen g) =>
  s -> (g -> s -> s) -> g -> DS.Stream s
mcmcP x0 kern g = DS.unfold stepUf (x0, g)
  where
    stepUf xg = let
      (x1, g1) = xg
      x2 = kern g1 x1
      (g2, _) = split g1
      in (x2, (x2, g2))
```
Calling these functions for our logistic regression example is similar to what we have seen before, but again there are minor syntactic differences depending on the approach. For further details see the full runnable scripts for the [monadic approach](https://github.com/darrenjw/logreg/blob/main/Haskell/lr/app/Rwmh.hs), the [pure sequential approach](https://github.com/darrenjw/logreg/blob/main/Haskell/lr/app/RwmhP.hs), and the [splittable approach](https://github.com/darrenjw/logreg/blob/main/Haskell/lr/app/RwmhPS.hs).

### Dex

Dex is a pure functional language and uses a splittable random number generator, so the style we use is similar to JAX (or Haskell using a splittable generator). We can generate a Metropolis kernel with
```haskell
def mKernel {s} (lpost: s -> Float) (rprop: Key -> s -> s) : 
    Key -> (s & Float) -> (s & Float) =
  def kern (k: Key) (sll: (s & Float)) : (s & Float) =
    (x0, ll0) = sll
    [k1, k2] = split_key k
    x = rprop k1 x0
    ll = lpost x
    a = ll - ll0
    u = rand k2
    select (log u < a) (x, ll) (x0, ll0)
  kern
```
We can then unfold our Markov chain with
```haskell
def markov_chain {s} (k: Key) (init: s) (kern: Key -> s -> s) (its: Nat) :
    Fin its => s =
  with_state init \st.
    for i:(Fin its).
      x = kern (ixkey k i) (get st)
      st := x
      x
```
Here we combine Dex's state effect with a for loop to unfold the stream. See the [full runnable script](https://github.com/darrenjw/logreg/blob/main/Dex/fit-bayes.dx) for further details.

## Next steps

As previously discussed, none of these codes are optimised, so care should be taken not to over-interpret running times. However, JAX and Dex are noticeably faster than the alternatives, even running on a single CPU core. Another interesting feature of both JAX and Dex is that they are *differentiable*. This makes it very easy to develop algorithms using *gradient* information. In subsequent posts we will think about the gradient of our example log-posterior and how we can use gradient information to develop "better" sampling algorithms.

The complete runnable scripts are all available from this [public github repo](https://github.com/darrenjw/logreg).



