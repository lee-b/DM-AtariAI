module Neural_Net where
import System.Random
import qualified Data.Array.Repa as R
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as VUN
import qualified Convolve as CV
import qualified Data.Array.Repa.Repr.Unboxed as RU
import qualified Data.Array.Repa.Algorithms.Matrix as RM
import qualified Data.Array.Repa.Algorithms.Randomish as RR
import qualified Configure as CF
import qualified Utils as U

-- Define the parameters for the two convulational layers
cnvLyr1Config = ([4, 84, 84],  16, [4, 8, 8],  4, U.sol [20, 20, 16, 1])
cnvLyr2Config = ([16, 20, 20], 32, [16, 4, 4], 2, U.sol [2592, 1])

type NNEnv = ((RU.Array RU.U R.DIM4 Double,
               RU.Array RU.U R.DIM4 Double, 
               RU.Array RU.U R.DIM2 Double, 
               RU.Array RU.U R.DIM2 Double),
              (RU.Array RU.U R.DIM4 Double,
               RU.Array RU.U R.DIM4 Double,
               RU.Array RU.U R.DIM2 Double,
               RU.Array RU.U R.DIM2 Double))
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
                    w = RR.randomishDoubleArray 
                          (U.sol $ reverse (numFltrs : fltrDim)) (-wBnd) wBnd 1
                    b = R.fromUnboxed ((U.sol [ftrMpSd, ftrMpSd, numFltrs, 1])) 
                          (VUN.replicate (numFltrs * ftrMpSd * ftrMpSd) 
                          (0 :: Double))
                in (w, b)
      -- Setup the weight and bias vector for layer 2
      (w2,b2) = let (inpImgDim, numFltrs, fltrDim, strd, _) = cnvLyr2Config
                    -- Feature map dimensions
                    ftrMpSd = (1 + quot (inpImgDim !! 1 - fltrDim !! 1) strd)
                    -- number of input and output connections per neuron
                    numInpPer = (fltrDim !! 0 * fltrDim !! 1 * fltrDim !! 2)
                    numOutPer =  (numFltrs * ftrMpSd * ftrMpSd)
                    wBnd = sqrt (6.0 / (fromIntegral (numInpPer + numOutPer)))
                    w = RR.randomishDoubleArray
                          (U.sol $ reverse (numFltrs : fltrDim)) (-wBnd) wBnd 1
                    b = R.fromUnboxed ((U.sol [ftrMpSd, ftrMpSd, numFltrs, 1])) 
                          (VUN.replicate (numFltrs * ftrMpSd * ftrMpSd)
                          (0 :: Double))
                in (w, b)
      -- Setup the weight and bias vector for layer 3
      (w3,b3) = let nIn = 32 * 9 * 9 -- Number of inputs
                    nOut = 256 -- Number of neurons
                    wBnd = sqrt (6.0 / (fromIntegral (nIn + nOut)))
                    w = RR.randomishDoubleArray ((U.sol [nOut, nIn])) (-wBnd)
                          wBnd 1
                    b = R.fromUnboxed ((U.sol [nOut, 1]))
                          (VUN.replicate nOut (0 :: Double))
                in (w, b)
      -- Setup the weight and bias vector for layer 4
      (w4,b4) = let nIn = 256 -- Number of inputs
                    nOut = length CF.availActns -- Number of neurons
                    wBnd = sqrt (6.0 / (fromIntegral (nIn + nOut)))
                    w = RR.randomishDoubleArray ((U.sol [nOut, nIn])) (-wBnd)
                          wBnd 1
                    b = R.fromUnboxed ((U.sol [nOut, 1]))
                          (VUN.replicate nOut (0 :: Double))
                in (w, b)
  in ((w1, w2, w3, w4), (b1, b2, b3, b4))


nnBestAction
  :: (Monad m)
  => V.Vector (VUN.Vector Double)
  -> NNEnv
  -> m ([Char])
nnBestAction mem nnEnv = do
  -- Link the layers and give output
  let ((w1, w2, w3, w4), (b1, b2, b3, b4)) = nnEnv
  if V.length mem < 4 then 
      return "0"
  else do
      actnProb <- evalActnProb nnEnv (getLastState mem)
        -- Get the most probable action
      return (CF.availActns !! (VUN.maxIndex actnProb))

evalActnProb nnEnv input = do
  -- Link the layers and give output
  let ((w1, w2, w3, w4), (b1, b2, b3, b4)) = nnEnv
  out1 <- cnvLyr cnvLyr1Config input w1 b1
  out2 <- cnvLyr cnvLyr2Config out1 w2 b2
  out3 <- cnctdLyr3 out2 w3 b3
  actnProb <- outptLyr4 out3 w4 b4
  return actnProb

nnTrain
  :: V.Vector (VUN.Vector Double)
  -> NNEnv
  -> IO NNEnv
nnTrain mem nnEnv = do
  let ((w1, w2, w3, w4), (b1, b2, b3, b4)) = nnEnv
      lMem = V.length mem
  g <- newStdGen

  -- Pick four random states from memory and train on them
  let indices = take 4 (randomRs (0, lMem) g :: [Int])
      states = map (getState mem) indices
      --actnProbs = map (evalActnProb nnEnv) states
      -- XXX fold with a helper here to acumulate an altered nnEnv at each step
  return initilaizeEnv -- "XXX"


getState mem n =
    -- Consturct the indices of the 4 frames and extract them from mem
    let indices = map (max 0) [n - 3..n]
        screens = map (mem V.!) indices
        as1DVector =  foldl (VUN.++) (head screens) (tail screens)
        asTensor = R.delay (R.fromUnboxed ((U.sol [84, 84, 4, 1]) :: R.DIM4)
                            as1DVector)
    in asTensor


getLastState mem =
    let last4 = V.take 4 mem
        as1DVector = V.foldl (VUN.++) (V.head last4) (V.tail last4)
        asTensor = R.delay (R.fromUnboxed (U.sol [84, 84, 4, 1]) as1DVector)
    in asTensor


cnvLyr lyrConfig input w b = do
  let (_, _, _, strd, outPShape) = lyrConfig
  convOutpt <- (CV.conv4D input (R.delay w) strd)
  let thresh = (0.0 :: Double)
      actvtn =  (R.+^) convOutpt b
      abvThresh = R.map (\e -> if e > thresh then (e - thresh) else 0) actvtn
      outP = R.reshape outPShape abvThresh
  return outP


cnctdLyr3 input w b = do
  -- input has extent 1, 32 * 9 * 9
  inputC <- R.computeUnboxedP input
  let actvtn = (R.+^) (RM.mmultS inputC w) b
      abvThresh = R.map (\e -> if e > 0.0 then (e - 0.0) else 0) actvtn
      outP = abvThresh
  -- outP has extent 256
  return outP 


outptLyr4 input w b = do
  -- input has extent 256
  inputC <- R.computeUnboxedP input
  let actvtn = (R.+^) (RM.mmultS inputC w) b
      outP = VUN.fromList (R.toList actvtn)
  -- outP has extent (length availActns)
  return outP
