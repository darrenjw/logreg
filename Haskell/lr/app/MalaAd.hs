{-# LANGUAGE ConstraintKinds, DataKinds, FlexibleContexts, GADTs,
 OverloadedStrings, PatternSynonyms, QuasiQuotes,
 ScopedTypeVariables, TemplateHaskell, TypeOperators, TypeApplications,
 ViewPatterns #-}

module MalaAd where

{-
***********************************************************************
***********************************************************************
PLEASE NOTE THAT THIS IS A NON-FUNCTIONAL WIP. THE STANDARD HASKELL
AUTO-DIFF LIBRARY CAN'T EASILY DIFFERENTIATE THROUGH MATRIX
OPERATIONS.
***********************************************************************
-}

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
rec2l :: Person -> [Double]
rec2l r = [1.0, fromIntegral $ rgetField @Npreg r, fromIntegral $ rgetField @Glu r,
           fromIntegral $ rgetField @Bp r, fromIntegral $ rgetField @Skin r,
            rgetField @Bmi r, rgetField @Ped r, fromIntegral $ rgetField @Age r]

-- sum an hmatrix Vector
vsum :: LA.Vector Double -> Double
vsum v = (LA.scalar 1) LA.<.> v

-- log-likelihood
ll :: (KnownNat n, KnownNat p) => L n p -> R n -> R p -> Double
ll x y b = (negate) (vsum (LA.cmap log (
                              (LA.scalar 1) + (LA.cmap exp (LA.cmap (negate) (
                                                         (((LA.scalar 2) * (extract y)) - (LA.scalar 1)) * (extract $ x #> b)
                                                         )
                                                     )))))

-- log-prior
pscale :: [Double] -- prior standard deviations
pscale = [10.0, 1, 1, 1, 1, 1, 1, 1]

lprior :: (KnownNat p) => R p -> Double
lprior b = sum $ (\x -> logDensity (normalDistr 0.0 (snd x)) (fst x)) <$> (zip (LA.toList $ extract b) pscale)
           
-- log-posterior
lpost :: (KnownNat n, KnownNat p) => L n p -> R n -> R p -> Double
lpost x y b = (ll x y b) + (lprior b)

-- MALA pre-conditioner
pre :: (KnownNat p) => R p -- relative scalings of the proposal noise
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
malaKernel :: (StatefulGen g m, KnownNat p) =>
  (R p -> Double) -> (R p -> R p) -> R p -> Double -> g ->
  (R p, Double) -> m (R p, Double)
malaKernel lpi glpi pre dt g = let
  sdt = sqrt dt
  spre = sqrt pre
  p = size pre
  advance beta = beta + (konst (0.5*dt))*pre*(glpi beta)
  rprop beta g = do
    zl <- (replicateM p . genContVar (normalDistr 0.0 1.0)) g
    let z = fromList zl
    return $ advance(beta) + (konst sdt)*spre*z
  dprop n o = let
    ao = advance o
    in sum $ (\i -> logDensity (normalDistr (extract ao LA.! i) ((extract spre LA.! i)*sdt)) (extract n LA.! i)) <$> [0..(p-1)]
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
  let xl = rec2l <$> F.toList dat
  let y = vector yl :: R 200
  print y
  let x = ll2m xl :: L 200 8
  disp 2 x
  -- AD tests
  --let glp = \b -> grVec $ grad (\bv -> (lpost x y (gvecR bv))) b :: R 8 -> R 8
  -- Do MCMC...
  --let b0 = vector [-9.0, 0, 0, 0, 0, 0, 0, 0] :: R 8
  --gen <- createSystemRandom
  --let kern = malaKernel (lpost x y) glp pre 1e-5 :: Gen RealWorld -> (R p, Double) -> IO (R p, Double)
  putStrLn "Running MALA with AD now..."
  --let ms = MS.drop burn $ mcmc (burn + its) th (b0, -1e50) kern gen
  --out <- MS.toList ms
  putStrLn "MCMC finished."
  --let mat = LA.fromLists (LA.toList <$> (fst <$> out))
  --LA.saveMatrix "malaAd.mat" "%g" mat
  putStrLn "*******************************************************************"
  putStrLn "*******************************************************************"
  putStrLn "Please note that this is a non-functional WIP. The standard Haskell auto-diff library can't easily differentiate through matrix operations."
  putStrLn "*******************************************************************"
  putStrLn "All done."


-- eof

