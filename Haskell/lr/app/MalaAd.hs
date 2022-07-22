{-# LANGUAGE ConstraintKinds, DataKinds, FlexibleContexts, GADTs,
 OverloadedStrings, PatternSynonyms, QuasiQuotes,
 ScopedTypeVariables, TemplateHaskell, TypeOperators, TypeApplications,
 ViewPatterns #-}

module MalaAd where

-- import Lib
import GHC.Prim
import Control.Monad
import qualified Data.Foldable as F
import Lens.Micro.Extras
import Frames
import Frames.TH (rowGen, RowGen(..))
import Pipes hiding (Proxy)
import Statistics.Distribution
import Statistics.Distribution.Normal
import Statistics.Distribution.Uniform
import System.Random
import System.Random.Stateful
import qualified System.Random.MWC as MWC
import qualified System.Random.MWC.Distributions as MWC
import qualified Data.Vector.Fusion.Stream.Monadic as MS
import Numeric.AD
import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra.Static
import Numeric.LinearAlgebra.Static.Vector
import GHC.TypeNats
import qualified Data.Vector.Sized as V

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
rec2v :: Person -> V.Vector 8 Double
rec2v r = V.fromTuple (1.0, fromIntegral $ rgetField @Npreg r, fromIntegral $ rgetField @Glu r,
           fromIntegral $ rgetField @Bp r, fromIntegral $ rgetField @Skin r,
            rgetField @Bmi r, rgetField @Ped r, fromIntegral $ rgetField @Age r)

-- generic casting
cast :: RealFloat a => Double -> a
cast x = fromRational $ toRational x

vcast :: (KnownNat p, RealFloat a) => V.Vector p Double -> V.Vector p a
vcast v = V.map cast v

-- dot product
vdot :: (KnownNat p, RealFloat a) => V.Vector p Double -> V.Vector p a -> a
vdot x y = V.sum $ (vcast x) * y

-- log-likelihood
ll :: RealFloat a => [V.Vector 8 Double] -> [Double] -> V.Vector 8 a -> a
ll x y b = sum $ (\z -> negate (log (1.0 + exp ((1.0-2.0*(cast (snd z)))*(vdot (fst z) b))))
                    ) <$> (zip x y)

-- log-prior
pscale :: RealFloat a => V.Vector 8 a -- prior standard deviations
pscale = V.fromTuple (10.0, 1, 1, 1, 1, 1, 1, 1)

lprior :: RealFloat a => V.Vector 8 a -> a
lprior b = negate $ V.sum $ (log pscale) + (0.5 * (b * b))
           
-- log-posterior
lpost :: RealFloat a => [V.Vector 8 Double] -> [Double] -> V.Vector 8 a -> a
lpost x y b = (ll x y b) + (lprior b)

-- MALA pre-conditioner
pre :: RealFloat a => V.Vector 8 a -- relative scalings of the proposal noise
pre = V.fromTuple (100.0, 1, 1, 1, 1, 1, 25, 1)

-- Metropolis-Hastings kernel
mhKernel :: (RealFloat a, StatefulGen g m) => (s -> a) -> (s -> g -> m s) ->
  (s -> s -> a) -> g -> (s, a) -> m (s, a)
mhKernel logPost rprop dprop g (x0, ll0) = do
  x <- rprop x0 g
  let ll = logPost(x)
  let ap = ll - ll0 + (dprop x0 x) - (dprop x x0)
  u <- (genContVar (uniformDistr 0.0 1.0)) g
  let next = if (cast (log u) < ap)
        then (x, ll)
        else (x0, ll0)
  return next

-- MALA kernel
malaKernel :: (StatefulGen g m, KnownNat p, RealFloat a) =>
  (V.Vector p a -> a) -> (V.Vector p a -> V.Vector p a) -> V.Vector p a -> a -> g ->
  (V.Vector p a, a) -> m (V.Vector p a, a)
malaKernel lpi glpi pre dt g = let
  sdt = sqrt dt
  spre = sqrt pre
  v = V.map (\vi -> dt * vi) pre
  d = length pre
  advance beta = beta + (V.map (\vi -> (0.5*dt)*vi) (pre*(glpi beta)))
  rprop beta g = do
    zl <- (replicateM d . genContVar (normalDistr 0.0 1.0)) g
    z <- sequence $ V.map (\vi -> genContVar (normalDistr 0.0 1.0) g) pre
    return $ advance(beta) + (V.map (\vi -> sdt*vi) (spre*(vcast z)))
  dprop n o = let
    ao = advance o
    diff = n - ao
    in -0.5*(sum ((log v) + (diff*diff/v)))
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

-- convert list of lists to matrix
ll2m :: (KnownNat n, KnownNat p) => [[Double]] -> L n p
ll2m xl = matrix $ concat xl


-- main entry point to this program
malaAd :: IO ()
malaAd = do
  putStrLn "Mala in Haskell, using auto-diff"
  let its = 10000 -- required number of iterations (post thinning and burn-in)
  let burn = 10 -- NB. This is burn-in AFTER thinning
  let th = 10 -- thinning interval
  -- read and process data
  dat <- loadData
  let yl = (\x -> if x then 1.0 else 0.0) <$> F.toList (view yy <$> dat)
  let xl = rec2v <$> F.toList dat
  print xl
  print yl
  -- AD tests
  let glp = grad (\b -> lpost xl yl b)
  -- Do MCMC...
  let b0 = V.fromTuple (-9.0, 0, 0, 0, 0, 0, 0, 0)
  gen <- MWC.createSystemRandom
  let kern = malaKernel (lpost xl yl) glp pre 1e-5 :: MWC.Gen RealWorld -> (V.Vector 8 Double, Double) -> IO (V.Vector 8 Double, Double)
  putStrLn "Running MALA with AD now..."
  let ms = MS.drop burn $ mcmc (burn + its) th (b0, -1e50) kern gen
  out <- MS.toList ms
  putStrLn "MCMC finished."
  let mat = LA.fromLists (V.toList <$> (fst <$> out))
  LA.saveMatrix "malaAd.mat" "%g" mat
  putStrLn "All done."


-- eof

