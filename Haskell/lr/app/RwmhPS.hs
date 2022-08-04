{-# LANGUAGE ConstraintKinds, DataKinds, FlexibleContexts, GADTs,
 OverloadedStrings, PatternSynonyms, QuasiQuotes,
 ScopedTypeVariables, TemplateHaskell, TypeOperators, TypeApplications,
 ViewPatterns #-}

-- RWMH with pure random numbers, using splitting, a la JAX and DEX

module RwmhPS where

import Rwmh

import GHC.Prim
import Control.Monad
import qualified Data.Foldable as F
import Lens.Micro.Extras
import Numeric.LinearAlgebra
--import Statistics.Distribution
import Statistics.Distribution.Normal
--import Statistics.Distribution.Uniform
import System.Random
--import System.Random.Stateful
--import System.Random.MWC
import qualified Data.Vector.Fusion.Stream.Monadic as MS
import qualified Data.Stream as DS

unif :: (RandomGen g) => g -> Double
unif g = fst $ uniformR (0.0, 1.0) g

stdNorm :: (RandomGen g) => g -> Double
stdNorm g = let
  (g1, g2) = split g
  u1 = unif g1
  u2 = unif g2
  th = u1*2*pi
  r2 = -2*(log u2)
  in (sqrt r2)*(sin th)

stdNorms :: (RandomGen g) => Int -> g -> [Double]
stdNorms n g = if (n == 0)
  then []
  else let
  (g1, g2) = split g
  z = stdNorm g1
  zs = stdNorms (n-1) g2
  in z : zs

-- Proposal (pure version)
rpropP :: (RandomGen g) =>  Vector Double -> g -> Vector Double
rpropP beta g = let
  p = size pre
  zl = stdNorms p g
  z = fromList zl
  in beta + 0.02 * pre * z

-- Metropolis kernel (pure version)
mKernelP :: (RandomGen g) => (s -> Double) -> (s -> g -> s) -> g -> (s, Double) -> (s, Double)
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

-- MCMC Stream (pure version)
mcmcP :: (RandomGen g) =>
  s -> (g -> s -> s) -> g -> DS.Stream s
mcmcP x0 kern g = DS.unfold stepUf (x0, g)
  where
    stepUf xg = let
      (x1, g1) = xg
      x2 = kern g1 x1
      (g2, _) = split g1
      in (x2, (x2, g2))


-- thin a Stream
thin :: Int -> DS.Stream s -> DS.Stream s
thin t xs = let
  xn = DS.drop t xs
  in DS.Cons (DS.head xn) (thin t xn)

-- main entry point to this program
rwmhPS :: IO ()
rwmhPS = do
  putStrLn "RWMH in Haskell (pure version, using splitting)"
  let its = 10000 -- required number of iterations (post thinning and burn-in)
  let burn = 1000 -- NB. This is burn-in BEFORE thinning
  let th = 1000 -- thinning interval
  -- read and process data
  dat <- loadData
  let yl = (\x -> if x then 1.0 else 0.0) <$> F.toList (view yy <$> dat)
  let xl = rec2l <$> F.toList dat
  let y = vector yl
  print y
  let x = fromLists xl
  disp 2 x
  -- Do MCMC...
  let b0 = fromList [-9.0, 0, 0, 0, 0, 0, 0, 0]
  gen <- initStdGen
  let kern = mKernelP (lpost x y) rpropP
  putStrLn "Running pure splitting RWMH now..."
  let ds = mcmcP (b0, -1e50) kern gen
  let out = DS.take its $ thin th $ DS.drop burn ds
  let mat = fromLists (toList <$> (fst <$> out))
  saveMatrix "rwmhPS.mat" "%g" mat
  putStrLn "MCMC finished."
  putStrLn "All done."


-- eof

