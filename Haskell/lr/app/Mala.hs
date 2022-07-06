{-# LANGUAGE ConstraintKinds, DataKinds, FlexibleContexts, GADTs,
 OverloadedStrings, PatternSynonyms, QuasiQuotes,
 ScopedTypeVariables, TemplateHaskell, TypeOperators, TypeApplications,
 ViewPatterns #-}

module Mala where

-- import Lib
import GHC.Prim
import Control.Monad
import qualified Data.Foldable as F
import Lens.Micro.Extras
import Frames
import Frames.TH (rowGen, RowGen(..))
import Pipes hiding (Proxy)
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

-- gradient
glp :: Matrix Double -> Vector Double -> Vector Double -> Vector Double
glp x y b = let
  glpr = -b / (fromList [100.0, 1, 1, 1, 1, 1, 1, 1])
  gll = (tr x) #> (y - (scalar 1)/((scalar 1) + (cmap exp (-x #> b))))
  in glpr + gll

-- MALA pre-conditioner
pre :: Vector Double -- relative scalings of the proposal noise
pre = fromList [100.0, 1, 1, 1, 1, 1, 25, 1]

-- Metropolis-Hastings kernel
mhKernel :: (StatefulGen g m) => (s -> Double) -> (s -> g -> m s) ->
  (s -> s -> Double) -> g -> (s, Double) -> m (s, Double)
mhKernel logPost rprop dprop g (x0, ll0) = do
  x <- rprop x0 g
  let ll = logPost(x)
  let a = ll - ll0 + (dprop x0 x) - (dprop x x0)
  u <- (genContVar (uniformDistr 0.0 1.0)) g
  let next = if ((log u) < a)
        then (x, ll)
        else (x0, ll0)
  return next

-- MALA kernel
malaKernel :: (StatefulGen g m) =>
  (Vector Double -> Double) -> (Vector Double -> Vector Double) -> Vector Double -> Double -> g ->
  (Vector Double, Double) -> m (Vector Double, Double)
malaKernel lpi glpi pre dt g = let
  sdt = sqrt dt
  spre = cmap sqrt pre
  p = size pre
  advance beta = beta + (scalar (0.5*dt))*pre*(glpi beta)
  rprop beta g = do
    zl <- (replicateM p . genContVar (normalDistr 0.0 1.0)) g
    let z = fromList zl
    return $ advance(beta) + (scalar sdt)*spre*z
  dprop n o = let
    ao = advance o
    in sum $ (\i -> logDensity (normalDistr (ao!i) ((spre!i)*sdt)) (n!i)) <$> [0..(p-1)]
  in mhKernel lpi rprop dprop g
  
-- MCMC stream
mcmc :: (StatefulGen g m) =>
  Int -> Int -> s -> (g -> s -> m s) -> g -> MS.Stream m s
mcmc it th x0 kern g = MS.iterateNM it (stepN th (kern g)) x0

-- Apply a monadic function repeatedly
stepN :: (Monad m) => Int -> (a -> m a) -> (a -> m a)
stepN n fa = if (n == 1)
  then fa
  else (\x -> (fa x) >>= (stepN (n-1) fa))


-- main entry point to this program
mala :: IO ()
mala = do
  putStrLn "Mala in Haskell"
  let its = 10000 -- required number of iterations (post thinning and burn-in)
  let burn = 10 -- NB. This is burn-in AFTER thinning
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
  gen <- createSystemRandom
  --pg <- initStdGen
  --gen <- newIOGenM pg
  let kern = malaKernel (lpost x y) (glp x y) pre 1e-5 :: Gen RealWorld -> (Vector Double, Double) -> IO (Vector Double, Double)
  putStrLn "Running MALA now..."
  let ms = MS.drop burn $ mcmc (burn + its) th (b0, -1e50) kern gen
  out <- MS.toList ms
  putStrLn "MCMC finished."
  let mat = fromLists (toList <$> (fst <$> out))
  saveMatrix "mala.mat" "%g" mat
  putStrLn "All done."


-- eof

