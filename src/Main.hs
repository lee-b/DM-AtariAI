-- Author: Hamzeh Alsalhi <ha258@cornell.edu>

module Main where

import Data.List.Split
import System.Posix.Files
import System.Posix.IO
import System.Random
import qualified Data.ByteString.Char8                as C
import qualified Data.Vector                          as V
import qualified Data.Vector.Unboxed                  as VUN
import qualified Neural_Net                           as NN
--import qualified Prelude                              as P
import qualified Configure                            as CF
import qualified Image_Processing                     as IP
import qualified ALE_Interface                        as ALE


main = do
  (toA, fromA) <- ALE.initialize
  -- Send action "noop, noop" to start game
  str <- C.hGetLine fromA
  fdWrite toA "1,18\n"
  -- Enter the main loop
  playGame fromA toA V.empty V.empty V.empty (0 :: Integer) (0 :: Int) (0 :: Int)
           NN.initilaizeEnv


playGame fromA toA screens actions rewards memSz frmsPlyd gamesPlayed nnEnv = 
 screens `seq` frmsPlyd `seq` memSz `seq` 
 do str <- C.hGetLine fromA
    let strTkns = C.split ':' str
        (scrStr, epInfo) = (strTkns !! 0, strTkns !! 1)
        reward = read (C.unpack $ (C.split ',' epInfo) !! 1) :: Integer
    
    if (C.split ',' epInfo) !! 0 == C.pack "1" then do
      putStrLn $ show epInfo
      fdWrite toA "45,1\n"
      playGame fromA toA screens actions rewards memSz frmsPlyd gamesPlayed nnEnv
    else do
      (act, _) <- (chooseAction screens actions rewards frmsPlyd nnEnv)
      nnEnvRet <- train         screens actions rewards frmsPlyd nnEnv
      fdWrite toA (act : ",18\n")
      let smallScr = (IP.scrnToNnInp scrStr) :: VUN.Vector Float
      putStrLn $ "Frames Played " ++ (show frmsPlyd)
      if memSz >= CF.memSize then
        playGame fromA toA (smallScr `V.cons` (V.init screens))
                           (act `V.cons` (V.init actions))
                           (reward `V.cons` (V.init rewards)) 
                 memSz(frmsPlyd + 1) gamesPlayed nnEnvRet
      else 
        playGame fromA toA (smallScr `V.cons` screens)
                           (act `V.cons` actions)
                           (reward `V.cons` rewards)
                           (memSz + 1) (frmsPlyd + 1)
          gamesPlayed nnEnvRet


train screens actions rewards frmsPlyd nnEnv = do
    nnEnvRet <- NN.nnTrain screens actions rewards nnEnv
    return nnEnvRet

chooseAction screens actions rewards frmsPlyd nnEnv = do
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
                          putStrLn ("Random Action " ++ [rndAct])
                          return rndAct
                  else do nnAct <- NN.nnBestAction screens nnEnv
                          --let nnAct = "0"
                          putStrLn ("Neural Network Choses " ++ [nnAct])
                          return nnAct
    action <- act
    return (action, nnEnv)
