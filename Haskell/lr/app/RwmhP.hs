{-# LANGUAGE ConstraintKinds, DataKinds, FlexibleContexts, GADTs,
 OverloadedStrings, PatternSynonyms, QuasiQuotes,
 ScopedTypeVariables, TemplateHaskell, TypeOperators, TypeApplications,
 ViewPatterns #-}

module RwmhP where

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

stdNorm :: (RandomGen g) => g -> (Double, g)
stdNorm g = let
  (u1, g1) = uniformR (0, 1) g
  (u2, g2) = uniformR (0, 1) g1
  th = u1*2*pi
  r2 = -2*(log u2)
  in ((sqrt r2)*(sin th), g2)

stdNorms :: (RandomGen g) => Int -> g -> ([Double], g)
stdNorms n g = if (n == 0)
  then ([], g)
  else let
  (z, g1) = stdNorm g
  (zs, g2) = stdNorms (n-1) g1
  in (z : zs, g2)

-- Proposal (pure version)
rpropP :: (RandomGen g) =>  Vector Double -> g -> (Vector Double, g)
rpropP beta g = let
  p = size pre
  (zl, g1) = stdNorms p g
  z = fromList zl
  in (beta + 0.02 * pre * z, g1)

-- Metropolis kernel (pure version)
mKernelP :: (RandomGen g) => (s -> Double) -> (s -> g -> (s, g)) -> g -> (s, Double) -> ((s, Double), g)
mKernelP logPost rprop g (x0, ll0) = let
  (x, g1) = rprop x0 g
  ll = logPost(x)
  a = ll - ll0
  (u, g2) = uniformR (0, 1) g1
  next = if ((log u) < a)
        then (x, ll)
        else (x0, ll0)
  in (next, g2)

-- MCMC Stream (pure version)
mcmcP :: (RandomGen g) => s -> (g -> s -> (s, g)) -> g -> DS.Stream s
mcmcP x0 kern g = DS.unfold stepUf (x0, g)
  where
    stepUf xg = let
      (x1, g1) = kern (snd xg) (fst xg)
      in (x1, (x1, g1))


-- thin a Stream
thin :: Int -> DS.Stream s -> DS.Stream s
thin t xs = let
  xn = DS.drop t xs
  in DS.Cons (DS.head xn) (thin t xn)

-- main entry point to this program
rwmhP :: IO ()
rwmhP = do
  putStrLn "RWMH in Haskell (pure version)"
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
  putStrLn "Running pure RWMH now..."
  let ds = mcmcP (b0, -1e50) kern gen
  let out = DS.take its $ thin th $ DS.drop burn ds
  let mat = fromLists (toList <$> (fst <$> out))
  saveMatrix "rwmhP.mat" "%g" mat
  putStrLn "MCMC finished."
  putStrLn "All done."


-- eof

