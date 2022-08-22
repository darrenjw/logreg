{-# LANGUAGE ConstraintKinds, DataKinds, FlexibleContexts, GADTs,
 OverloadedStrings, PatternSynonyms, QuasiQuotes,
 ScopedTypeVariables, TemplateHaskell, TypeOperators, TypeApplications,
 ViewPatterns #-}

module Lang where

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

-- gradient
glp :: Matrix Double -> Vector Double -> Vector Double -> Vector Double
glp x y b = let
  glpr = -b / (fromList [100.0, 1, 1, 1, 1, 1, 1, 1])
  gll = (tr x) #> (y - (scalar 1)/((scalar 1) + (cmap exp (-x #> b))))
  in glpr + gll

-- MALA pre-conditioner
pre :: Vector Double -- relative scalings of the proposal noise
pre = fromList [100.0, 1, 1, 1, 1, 1, 25, 1]
  
-- Unadjusted Langevin kernel
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
lang :: IO ()
lang = do
  putStrLn "Unadjusted Langevin in Haskell"
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
  print (glp x y b0)
  -- prepare for MCMC
  gen <- createSystemRandom
  let kern = ulKernel (glp x y) pre 1e-6 :: Gen RealWorld -> Vector Double -> IO (Vector Double)
  putStrLn "Running Langevin now..."
  let ms = MS.drop burn $ mcmc (burn + its) th b0 kern gen
  out <- MS.toList ms
  putStrLn "MCMC finished."
  let mat = fromLists (toList <$> out)
  saveMatrix "lang.mat" "%g" mat
  putStrLn "All done."


-- eof

