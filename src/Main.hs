-- Author: Hamzeh Alsalhi <93hamsal@gmail.com>
module Main where
import Data.List.Split


import System.Posix.Files
import System.Posix.IO
import System.Random

import qualified Data.ByteString.Char8 as C
import qualified Data.Vector as V
import qualified Neural_Net as NN
import qualified Configure as CF
import qualified Image_Processing as IP
import qualified ALE_Interface as ALE


main = do
  (toA, fromA) <- ALE.initialize
  -- Send action "noop, noop" to start game
  str <- C.hGetLine fromA
  fdWrite toA "1,18\n"
  -- Enter the main loop
  playGame fromA toA V.empty 0 0 0 NN.initilaizeEnv


playGame fromA toA mem memSz frmsPlyd gamesPlayed nnEnv = 
 mem `seq` frmsPlyd `seq` memSz `seq` 
 do str <- C.hGetLine fromA
    let strTkns = C.split ':' str
    let (scrStr, epInfo) = (strTkns !! 0, strTkns !! 1)
    
    if (C.split ',' epInfo) !! 0 == C.pack "1" then do
      putStrLn $ show epInfo
      fdWrite toA "45,1\n"
      playGame fromA toA mem memSz frmsPlyd gamesPlayed nnEnv
    else do
      (act, nnEnvRet) <- (chooseAction mem frmsPlyd nnEnv)
      fdWrite toA (act ++ ",18\n")
      let smallScr = IP.scrnToNnInp scrStr
      putStrLn $ "Frames Played " ++ (show frmsPlyd)
      if memSz >= CF.memSize then
        playGame fromA toA (smallScr `V.cons` (V.init mem)) memSz (frmsPlyd + 1) 
          gamesPlayed nnEnvRet
      else 
        playGame fromA toA (smallScr `V.cons` mem) (memSz + 1) (frmsPlyd + 1)
          gamesPlayed nnEnvRet


chooseAction mem frmsPlyd nnEnv = do
    -- Random number generator
    g <- getStdGen
    let (rndRl,_) = randomR (0.0, 1.0) g
    g <- newStdGen
    let (rndIdx,_) = randomR (0, (length CF.availActns) - 1) g

    let epsilon = max ((0.9 :: Float) - fromIntegral frmsPlyd / 1000000.0) 0.1
    putStrLn $ "Epsilon " ++ show epsilon
    let actRandomly = rndRl < epsilon
        act = do
                if actRandomly 
                  then do let rndAct = (CF.availActns !! rndIdx)
                          putStrLn ("Random Action " ++ rndAct)
                          return rndAct
                  else do nnAct <- NN.nnBestAction mem nnEnv
                          --let nnAct = "0"
                          putStrLn ("Neural Network Choses " ++ nnAct)
                          return nnAct
    action <- act
    return (action, nnEnv)
