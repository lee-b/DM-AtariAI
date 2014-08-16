module Neural_Net where
import qualified Data.Array.Repa as R
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as VUN
import qualified Convolve as CV
import qualified Data.Array.Repa.Repr.Unboxed as RU
import qualified Data.Array.Repa.Algorithms.Matrix as RM
import qualified Data.Array.Repa.Algorithms.Randomish as RR
import Debug.Trace
import qualified Configure as CF

debug = flip trace
-- use: variable `debug` "at variable"

strct = flip seq

assert_eq :: (Eq a, Show a) => a -> a -> [Char] -> Bool
assert_eq x y mrkrMsg =
  let c = if x == y then True  else (error (mrkrMsg ++ show x ++ show y ++ "are not equal!"))
  in c `strct` c

sol :: R.Shape sh => [Int] -> sh
sol = R.shapeOfList

los :: R.Shape sh => sh -> [Int]
los = R.listOfShape

nnBestAction
  :: (Monad m)
  => V.Vector (VUN.Vector Double)
  -> m ([Char])

-- Define the parameters for the two convulational layers
cnvLyr1Config = ([4, 84, 84],
                 16,
                 [4, 8, 8],
                 4,
                 sol [20, 20, 16, 1])

cnvLyr2Config =  ([16, 20, 20],
                  32,
                  [16, 4, 4],
                  2,
                  sol [2592, 1])

nnBestAction mem = do
  -- Main neural netowrk linking layers will be implemented here
  if V.length mem < 4 then 
      return "0"
  else do
      let rcnt4 = (V.take 4 mem)
       -- Stitch last 4 images together into 4D tensor
      let rcnt = (V.foldl (VUN.++) (V.head rcnt4) (V.tail rcnt4))
      let tens = (R.delay (R.fromUnboxed (sol [84, 84, 4, 1]) rcnt))
            -- Send input to first layer and propogate through to output layer
      out1 <- cnvLyr cnvLyr1Config tens
      out2 <- cnvLyr cnvLyr2Config out1
      out3 <- cnctdLyr3 out2
      actnProb <- outptLyr4 out3
        -- Get the most probable action
      return (CF.availActns !! (VUN.maxIndex actnProb))
-- ##

cnvLyr lyrConfig input = do
  let (inpImgDim, numFltrs, fltrDim, strd, outPShape) = lyrConfig
      -- number of input connections per neuron
      numInpPer =  (fltrDim!!0 * fltrDim!!1 * fltrDim!!2)
      -- Feature map dimensions
      ftrMpSd = (1 + quot (inpImgDim!!1 - fltrDim!!1) strd)
      ftrMpDim = [ftrMpSd, ftrMpSd]
      -- number of input connections per neuron
      numOutPer =  (numFltrs * ftrMpDim!!0 * ftrMpDim!!1)
      wBnd = sqrt (6.0 / (fromIntegral (numInpPer + numOutPer)))
      w = RR.randomishDoubleArray (sol $ reverse (numFltrs : fltrDim)) (-wBnd) wBnd 1
      -- XXX on Neural Netowkr update b should be a list of numFltrs value each replicated ftrMapSd * ftrMpSd times
      b = [0 | _ <- [1..numFltrs * ftrMpSd * ftrMpSd]]
      b_tens = RU.fromListUnboxed (sol [ftrMpSd, ftrMpSd, numFltrs, 1]) b
  convOutpt <- (CV.conv4D input (R.delay w) strd)
  let thresh = (0.0 :: Double)
      actvtn =  (R.+^) convOutpt b_tens
      abvThresh = R.map (\e -> if e > thresh then (e - thresh) else 0) actvtn
      outP = R.reshape outPShape abvThresh
  return outP

cnctdLyr3 input = do
  -- input has extent 1, 32 * 9 * 9
  let nIn = 32 * 9 * 9 -- Number of inputs
      nOut = 256 -- Number of neurons
      wBnd = sqrt (6.0 / (fromIntegral (nIn + nOut)))
      w = RR.randomishDoubleArray (sol [nOut, nIn]) (-wBnd) wBnd 1
      b = R.fromUnboxed (sol [nOut, 1]) (VUN.replicate nOut (0 :: Double))
      thresh = 0.0
  inputC <- R.computeUnboxedP input
  let actvtn = (R.+^) (RM.mmultS inputC w) b
      abvThresh = R.map (\e -> if e > thresh then (e - thresh) else 0) actvtn
      outP = abvThresh
  -- outP has extent 256
  return outP 

outptLyr4 input= do
  -- input has extent 256
  let nIn = 256 -- Number of inputs
      nOut = length CF.availActns -- Number of neurons
      wBnd = sqrt (6.0 / (fromIntegral (nIn + nOut)))
      w = RR.randomishDoubleArray (sol [nOut, nIn]) (-wBnd) wBnd 1
      b = R.fromUnboxed (sol [nOut, 1]) (VUN.replicate nOut (0 :: Double))
  inputC <- R.computeUnboxedP input
  let actvtn = (R.+^) (RM.mmultS inputC w) b
      outP = VUN.fromList (R.toList actvtn)
  -- outP has extent (length availActns)
  return outP