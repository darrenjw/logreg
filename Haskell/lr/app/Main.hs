{-# LANGUAGE ConstraintKinds, DataKinds, FlexibleContexts, GADTs,
 OverloadedStrings, PatternSynonyms, QuasiQuotes,
 ScopedTypeVariables, TemplateHaskell, TypeOperators, TypeApplications,
 ViewPatterns #-}

module Main where

-- import Lib
import GHC.Prim
import Control.Monad
import qualified Control.Foldl as L
import Control.Arrow ((&&&))
import qualified Data.Foldable as F
import Data.Vinyl
import Data.Vinyl.Functor (Identity(..), Const(..))
import Lens.Micro
import Lens.Micro.Extras
import Frames
import Frames.TH (rowGen, RowGen(..))
import Pipes hiding (Proxy)
import qualified Pipes as P
import qualified Pipes.Prelude as P
import Numeric.LinearAlgebra
import Statistics.Distribution
import Statistics.Distribution.Normal
import Statistics.Distribution.Uniform
import System.Random
import System.Random.Stateful
import System.Random.MWC
import qualified System.Random.MWC.Distributions as MWC
import qualified Data.Vector.Fusion.Stream.Monadic as MS

-- template Haskell to create the Person type, and personParser
tableTypes' (rowGen "../../pima.data")
            { rowTypeName = "Person"
            , columnNames = [ "npreg", "glu", "bp"
                            , "skin", "bmi", "ped", "age", "yy" ]
            , separator = " " }

-- create a data stream
dataStream :: MonadSafe m => Producer Person m ()
dataStream = readTableOpt personParser "../../pima.data"

-- load full dataset
loadData :: IO (Frame Person)
loadData = inCoreAoS dataStream

-- create rows of covariate matrix
rec2l :: Person -> [Double]
rec2l r = [1.0, fromIntegral $ rgetField @Npreg r, fromIntegral $ rgetField @Glu r,
           fromIntegral $ rgetField @Bp r, fromIntegral $ rgetField @Skin r,
            rgetField @Bmi r, rgetField @Ped r, fromIntegral $ rgetField @Age r]

-- sum an hmatrix Vector
vsum :: Vector Double -> Double
vsum v = (konst 1 (size v) :: Vector Double) <.> v

-- log-likelihood
ll :: Matrix Double -> Vector Double -> Vector Double -> Double
ll x y b = (negate) (vsum (cmap log (
                              (scalar 1) + (cmap exp (cmap (negate) (
                                                         (((scalar 2) * y) - (scalar 1)) * (x #> b)
                                                         )
                                                     )))))

-- log-prior
pscale :: [Double] -- prior standard deviations
pscale = [10.0, 1, 1, 1, 1, 1, 1, 1]

lprior :: Vector Double -> Double
lprior b = sum $ (\x -> logDensity (normalDistr 0.0 (snd x)) (fst x)) <$> (zip (toList b) pscale)
           
-- log-posterior
lpost :: Matrix Double -> Vector Double -> Vector Double -> Double
lpost x y b = (ll x y b) + (lprior b)

-- symmetric proposal function
pre :: Vector Double -- relative scalings of the proposal noise
pre = fromList [10.0, 1, 1, 1, 1, 1, 5, 1]

rprop :: (StatefulGen g m) =>  Vector Double -> g -> m (Vector Double)
rprop beta g = do
  let p = size pre
  zl <- (replicateM p . genContVar (normalDistr 0.0 0.02)) g
  let z = fromList zl
  return (beta + pre * z)

-- Metropolis kernel
mhKernel :: (StatefulGen g m) => (s -> Double) -> (s -> g -> m s) -> g -> (s, Double) -> m (s, Double)
mhKernel logPost rprop g (x0, ll0) = do
  x <- rprop x0 g
  let ll = logPost(x)
  let a = ll - ll0
  u <- (genContVar (uniformDistr 0.0 1.0)) g
  let next = if ((log u) < a)
        then (x, ll)
        else (x0, ll0)
  return next

-- MCMC stream
mcmc :: (StatefulGen g m) =>
  (s, Double) -> (g -> (s, Double) -> m (s, Double)) -> g -> [m (s, Double)]
mcmc x0 kern g = iterate (\mx -> mx >>= (kern g)) (kern g x0)

-- thin a (lazy) list
thin :: Int -> [s] -> [s]
thin t xs = let
  xn = drop t xs
  in if (null xn)
    then []
    else (head xn) : (thin t xn)




-- main entry point to the program
main :: IO ()
main = do
  putStrLn "RWMH in Haskell"
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
  gen <- createSystemRandom
  let kern = mhKernel (lpost x y) rprop :: Gen RealWorld -> (Vector Double, Double) -> IO (Vector Double, Double)
  let sm = mcmc (b0, -1e50) kern gen
  putStrLn "Running RWMH now..."
  let smt = take 10000 $ thin 2 $ drop 100 sm
  putStrLn "Sequencing..."
  st <- sequence smt
  putStrLn "MCMC finished."
  let m = fromLists (toList <$> (fst <$> st))
  --disp 2 m
  saveMatrix "rwmh.mat" "%g" m
  putStrLn "All done."




-- eof

