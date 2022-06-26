# Bayesian logistic regression in Haskell


The Haskell code builds using [stack](https://docs.haskellstack.org/en/stable/README/). `stack` is the [Haskell](https://www.haskell.org/) equivalent of Scala's `sbt`. `stack` should manage all necessary dependencies, including Haskell itself. Note that on Linux (and similar) systems, Haskell and Stack can be installed by installing the packages `haskell-platform` and `haskell-stack`.


```bash
stack build && stack exec lr-exe
```

For convenience, a `Makefile` is also included, so just typing `make` should work if `stack` is installed.
