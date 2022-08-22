
module Main where

import Rwmh
import RwmhP
import RwmhPS
import Lang
import Mala
import MalaAd
import Hmc

import System.Environment
import System.Exit

main :: IO ()
main = do
  args <- getArgs
  let l = length args
  case l of
    0 -> do
      putStrLn "Choices: rwmh rwmhP rwmhPS lang mala (malaAd) hmc"
      exitFailure
    1 -> do
      let choice = head args
      case choice of
        "rwmh" -> rwmh
        "rwmhP" -> rwmhP
        "rwmhPS" -> rwmhPS
        "lang" -> lang
        "mala" -> mala
        "malaAd" -> malaAd
        "hmc" -> hmc

