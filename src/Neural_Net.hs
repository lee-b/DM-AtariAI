module Neural_Net where

import System.Random
import qualified System.Posix.Types as P
import qualified GHC.IO.Handle.Types as T
import Data.Dynamic
import qualified Data.Array.Repa                                      as R
import qualified Data.Array.Repa.Eval                                 as R
import qualified Data.Array.Repa.Unsafe                               as R
import qualified Data.Vector                                          as V
import qualified Data.Vector.Unboxed                                  as VUN
--import qualified ConvolveCPU                                          as CV
--import qualified ConvolveGPU                                          as CV
import qualified ConvolvePY                                           as CV
import qualified Data.Array.Repa.Repr.Unboxed                         as RU
import qualified Data.Array.Repa.Algorithms.Matrix                    as RM
import qualified Data.Array.Repa.Algorithms.Randomish                 as RR
import qualified Configure                                            as CF
import qualified Utils                                                as U

-- Define the parameters for the two convulational layers
cnvLyr1Config = ([4, 84, 84],  16, [4, 8, 8],  4, U.sol [20, 20, 16, 1])
cnvLyr2Config = ([16, 20, 20], 32, [16, 4, 4], 2, U.sol [2592, 1])

type NNEnv = ((RU.Array RU.U R.DIM4 Float,
               RU.Array RU.U R.DIM4 Float, 
               RU.Array RU.U R.DIM2 Float, 
               RU.Array RU.U R.DIM2 Float),
              (RU.Array RU.U R.DIM4 Float,
               RU.Array RU.U R.DIM4 Float,
               RU.Array RU.U R.DIM2 Float,
               RU.Array RU.U R.DIM2 Float))
initilaizeEnv :: NNEnv
initilaizeEnv =
  -- Setup the weight and bias vector for layer 1
  let (w1,b1) = let (inpImgDim, numFltrs, fltrDim, strd, _) = cnvLyr1Config
                    -- Feature map dimensions
                    ftrMpSd = (1 + quot (inpImgDim !! 1 - fltrDim !! 1) strd)
                    -- number of input and output connections per neuron
                    numInpPer = (fltrDim !! 0 * fltrDim !! 1 * fltrDim !! 2)
                    numOutPer =  (numFltrs * ftrMpSd * ftrMpSd)
                    wBnd = sqrt (6.0 / (fromIntegral (numInpPer + numOutPer)))
                    dummyWieghtTest = R.fromListUnboxed 
                                        (U.sol $ reverse (numFltrs : fltrDim))
                                        [0.037 | 
                                         _ <-  [1..numFltrs * product fltrDim]]
                    w = dummyWieghtTest
                    -- XXX reactivate this
                    --w = RR.randomishDoubleArray 
                    --      (U.sol $ reverse (numFltrs : fltrDim))
                    --        (-wBnd)
                    --        wBnd
                    --        1
                    b = R.fromUnboxed ((U.sol [ftrMpSd, ftrMpSd, numFltrs, 1])) 
                          (VUN.replicate (numFltrs * ftrMpSd * ftrMpSd) 
                          (0 :: Float))
                in (w, b)
      -- Setup the weight and bias vector for layer 2
      (w2,b2) = let (inpImgDim, numFltrs, fltrDim, strd, _) = cnvLyr2Config
                    -- Feature map dimensions
                    ftrMpSd = (1 + quot (inpImgDim !! 1 - fltrDim !! 1) strd)
                    -- number of input and output connections per neuron
                    numInpPer = (fltrDim !! 0 * fltrDim !! 1 * fltrDim !! 2)
                    numOutPer =  (numFltrs * ftrMpSd * ftrMpSd)
                    wBnd = sqrt (6.0 / (fromIntegral (numInpPer + numOutPer)))
                    dummyWieghtTest = R.fromListUnboxed 
                                        (U.sol $ reverse (numFltrs : fltrDim))
                                        [0.037 |
                                         _ <- [1..numFltrs * product fltrDim]]
                    w = dummyWieghtTest
                    -- XXX reactivate this
                    --w = RR.randomishDoubleArray
                    --      (U.sol $ reverse (numFltrs : fltrDim))
                    --        (-wBnd) 
                    --        wBnd 
                    --        1
                    b = R.fromUnboxed ((U.sol [ftrMpSd, ftrMpSd, numFltrs, 1])) 
                          (VUN.replicate (numFltrs * ftrMpSd * ftrMpSd)
                          (0 :: Float))
                in (w, b)
      -- Setup the weight and bias vector for layer 3
      (w3,b3) = let nIn = 32 * 9 * 9 -- Number of inputs
                    nOut = 256 -- Number of neurons
                    wBnd = sqrt (6.0 / (fromIntegral (nIn + nOut)))
                    dummyWieghtTest = R.fromListUnboxed 
                                        (U.sol [nOut, nIn])
                                        [0.037 | _ <-  [1..nOut * nIn]]
                    w = dummyWieghtTest
                    --w = RR.randomishDoubleArray ((U.sol [nOut, nIn])) (-wBnd)
                    --      wBnd 1
                    b = R.fromUnboxed ((U.sol [nOut, 1]))
                          (VUN.replicate nOut (0 :: Float))
                in (w, b)
      -- Setup the weight and bias vector for layer 4
      (w4,b4) = let nIn = 256 -- Number of inputs
                    nOut = length CF.availActns -- Number of neurons
                    wBnd = sqrt (6.0 / (fromIntegral (nIn + nOut)))
                    dummyWieghtTest = R.fromListUnboxed 
                                        (U.sol [nOut, nIn])
                                        [0.037 | _ <-  [1..nOut * nIn]]
                    w = dummyWieghtTest
                    --w = RR.randomishDoubleArray ((U.sol [nOut, nIn])) (-wBnd)
                    --      wBnd 1
                    b = R.fromUnboxed ((U.sol [nOut, 1]))
                          (VUN.replicate nOut (0 :: Float))
                in (w, b)
  in ((w1, w2, w3, w4), (b1, b2, b3, b4))


nnBestAction
  :: T.Handle 
  -> T.Handle
  -> V.Vector (VUN.Vector Float)
  -> NNEnv
  -> IO (Char)
nnBestAction toC fromC screens nnEnv = do
  -- Link the layers and give output
  let ((w1, w2, w3, w4), (b1, b2, b3, b4)) = nnEnv
  if V.length screens < 4 then 
      return '0'
  else do
      actnProb <- evalActnProb toC fromC nnEnv (getLastState screens)
        -- Get the most probable action
      return (CF.availActns !! (VUN.maxIndex actnProb))

--evalActnProb 
--  :: (Monad m) 
--  => NNEnv 
--  -> RU.Array R.D R.DIM4 Float 
--  -> m(VUN.Vector Float)
evalActnProb toC fromC nnEnv input = do
  -- Link the layers and give output
  let ((w1, w2, w3, w4), (b1, b2, b3, b4)) = nnEnv
  out1 <- cnvLyr toC fromC cnvLyr1Config input w1 b1
  out2 <- cnvLyr toC fromC cnvLyr2Config out1 w2 b2
  out3 <- cnctdLyr3 out2 w3 b3
  actnProb <- outptLyr4 out3 w4 b4
  return actnProb


trainHelper
  :: T.Handle
  -> T.Handle
  -> IO (NNEnv)
  -> (RU.Array R.D R.DIM4 Float, Char, Integer, RU.Array R.D R.DIM4 Float)
  -> IO (NNEnv)
trainHelper toC fromC nnEnv transition = do
    let (pre, act, reward, post) = transition
    nnEnvUnwraped <- nnEnv
    -- Train the nn on transition here
    preProb <- evalActnProb toC fromC nnEnvUnwraped pre
    postProb <- evalActnProb toC fromC nnEnvUnwraped post

    return nnEnvUnwraped


nnTrain
  :: T.Handle
  -> T.Handle
  -> V.Vector (VUN.Vector Float)
  -> V.Vector Char
  -> V.Vector Integer
  -> NNEnv
  -> IO NNEnv
nnTrain toC fromC screens actions rewards nnEnv = do
  let ((w1, w2, w3, w4), (b1, b2, b3, b4)) = nnEnv
      lMem = V.length screens
  g <- newStdGen

  if V.length screens < 4 then do
    return nnEnv
  else do
    -- Pick n random states from memory and train on them
    let numMiniBatches = 32
        indices = take numMiniBatches (randomRs (0, lMem - 2) g :: [Int])
        getTransition i = ((getState screens i), actions V.! i, rewards V.! i, (getState screens (i + 1)))
        transitions = map getTransition indices
    putStrLn ("Training on state indices: " ++ show indices)

    -- Our fold helper that accumulates on nnEnv
    nnNew <- (foldl (trainHelper toC fromC) (return nnEnv) transitions)
    return nnNew


getState screens n =
    -- Consturct the indices of the 4 frames and extract them from mem
    let indices = map (max 0) [n - 3..n]
        screenState = map (screens V.!) indices
        as1DVector =  foldl (VUN.++) (head screenState) (tail screenState)
        asTensor = R.delay (R.fromUnboxed ((U.sol [84, 84, 4, 1]) :: R.DIM4)
                            as1DVector)
    in asTensor


getLastState screens =
    let last4 = V.take 4 screens
        as1DVector = V.foldl (VUN.++) (V.head last4) (V.tail last4)
        asTensor = R.delay (R.fromUnboxed (U.sol [84, 84, 4, 1]) as1DVector)
    in asTensor

--cnvLyr 
--  :: (Monad m, R.Shape sh1, R.Shape sh2) 
--  => ([Int], Int, [Int], Int, sh1)
--  -> RU.Array R.D R.DIM4 Float 
--  -> RU.Array R.U R.DIM4 Float 
--  -> RU.Array R.U R.DIM4 Float
--  -> m(sh2)
cnvLyr toC fromC lyrConfig input w b = do
  let (_, _, _, strd, outPShape) = lyrConfig
  convOutpt <- (CV.conv4D toC fromC input (R.delay w) strd)
  let thresh = (0.0 :: Float)
      actvtn =  (R.+^) convOutpt b
      abvThresh = R.map (\e -> if e > thresh then (e - thresh) else 0) actvtn
      outP = R.reshape outPShape abvThresh
  return outP

--cnctdLyr3
--  :: (Monad m) 
--  => RU.Array R.D R.DIM2 Float 
--  -> RU.Array R.U R.DIM2 Float 
--  -> RU.Array R.U R.DIM2 Float
--  -> m(RU.Array R.D R.DIM2 Float)
cnctdLyr3 input w b = do
  -- input has extent 1, 32 * 9 * 9
  inputC <- R.computeUnboxedP input
  prodIW <- mmultP inputC w
  let actvtn = (R.+^) prodIW b
      abvThresh = R.map (\e -> if e > 0.0 then (e - 0.0) else 0) actvtn
      outP = abvThresh
  -- outP has extent 256
  return outP 

--outptLyr4
--  :: (Monad m) 
--  => RU.Array R.D R.DIM2 Float 
--  -> RU.Array R.U R.DIM2 Float 
--  -> RU.Array R.U R.DIM2 Float
--  -> m(VUN.Vector Float)
outptLyr4 input w b = do
  -- input has extent 256
  inputC <- R.computeUnboxedP input
  prodIW <- mmultP inputC w
  let actvtn = (R.+^) prodIW b
      outP = VUN.fromList (R.toList actvtn)
  -- outP has extent (length availActns)
  return outP

mmultP  :: Monad m
        => R.Array RU.U R.DIM2 Float 
        -> R.Array RU.U R.DIM2 Float 
        -> m (R.Array RU.U R.DIM2 Float)

mmultP arr brr 
 = [arr, brr] `R.deepSeqArrays` 
   do   trr      <- transpose2P brr
        let (R.Z R.:. h1  R.:. _)  = R.extent arr
        let (R.Z R.:. _   R.:. w2) = R.extent brr
        R.computeP 
         $ R.fromFunction (R.Z R.:. h1 R.:. w2)
         $ \ix   -> R.sumAllS 
                  $ R.zipWith (*)
                        (R.unsafeSlice arr (R.Any R.:. (RM.row ix) R.:. R.All))
                        (R.unsafeSlice trr (R.Any R.:. (RM.col ix) R.:. R.All))

transpose2P
        :: Monad m 
        => R.Array RU.U R.DIM2 Float 
        -> m (R.Array RU.U R.DIM2 Float)

transpose2P arr
 = arr `R.deepSeqArray`
   do   R.computeUnboxedP 
         $ R.unsafeBackpermute new_extent swap arr
 where  swap (R.Z R.:. i R.:. j)      = R.Z R.:. j R.:. i
        new_extent              = swap (R.extent arr)