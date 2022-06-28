
module Main where

import Rwmh
import RwmhP

import System.Environment
import System.Exit

main :: IO ()
main = do
  args <- getArgs
  let l = length args
  case l of
    0 -> do
      putStrLn "Choices: rwmh rwmhP"
      exitFailure
    1 -> do
      let choice = head args
      case choice of
        "rwmh" -> rwmh
        "rwmhP" -> rwmhP

