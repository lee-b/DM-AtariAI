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

nnBestAction
  :: (Monad m)
  => V.Vector (VUN.Vector Double)
  -> m ([Char])

-- Define the parameters for the two convulational layers
cnvLyr1Config = ([4 :: Int, 84 :: Int, 84 :: Int],       -- image dimensions
                 16 :: Int,                              -- num filters
                 [4 :: Int, 8 :: Int, 8 :: Int],         -- filter dimensions
                 4 :: Int,                               -- stride
                 (R.Z R.:. (1::Int) R.:. (16::Int) 
                  R.:. (20::Int) R.:. (20::Int)))        -- output shape

cnvLyr2Config =  ([16 :: Int, 20 :: Int, 20 :: Int],     -- image dimensions
                  32 :: Int,                             -- num filters
                  [16 :: Int, 4 :: Int, 4 :: Int],       -- filter dimensions
                  2 :: Int,                              -- stride
                  (R.Z R.:. (1::Int) R.:. (2592::Int)))  -- output shape

nnBestAction mem = do
  -- Main neural netowrk linking layers will be implemented here
  if V.length mem < 4 then 
      return "0"
  else do
      let rcnt4 = (V.take 4 mem) `strct` mem
       -- Stitch last 4 images together into 4D tensor
      let rcnt = (V.foldl (VUN.++) (V.head rcnt4) (V.tail rcnt4)) `strct` rcnt4
      let tens = (R.delay (R.fromUnboxed (R.Z R.:. (1 :: Int) R.:. (4 :: Int) R.:. (84 :: Int) R.:. (84 :: Int)) rcnt)) `strct` rcnt
            -- Send input to first layer and propogate through to output layer
      out1 <- cnvLyr cnvLyr1Config tens
      out2 <- cnvLyr cnvLyr2Config out1
      out3 <- cnctdLyr3 out2
      actnProb <- outptLyr4 out3
        -- Get the most probable action
      return (CF.availActns!!(VUN.maxIndex actnProb) `strct` actnProb)
-- ##

cnvLyr lyrConfig input = do
  let (inpImgDim, numFltrs, fltrDim, strd, outPShape) = lyrConfig
      -- number of input connections per neuron
      numInpPer =  ((fltrDim!!0 * fltrDim!!1 * fltrDim!!2) :: Int)
      -- Feature map dimensions
      --ftrMpSd = (1 + (round ((fromIntegral (inpImgDim!!1 - fltrDim!!1)) / strd)) :: Int) `strct` numInpPer `debug` "ftrMpSd"
      ftrMpSd = ((1 + quot (inpImgDim!!1 - fltrDim!!1) strd) :: Int)
      ftrMpDim = [ftrMpSd, ftrMpSd]
      -- number of input connections per neuron
      numOutPer =  ((numFltrs * ftrMpDim!!0 * ftrMpDim!!1) :: Int)
      wBnd = sqrt (6.0 / (fromIntegral (numInpPer + numOutPer)))
      w = RR.randomishDoubleArray (R.Z R.:. (numFltrs::Int) R.:. ((fltrDim!!0)::Int) R.:. ((fltrDim!!1)::Int) R.:. ((fltrDim!!2)::Int)) (-wBnd) wBnd 1
      -- XXX on Neural Netowkr update b should be a list of numFltrs value each replicated ftrMapSd * ftrMpSd times
      b = [0 | _ <- [1..numFltrs * ftrMpSd * ftrMpSd]]
      b_tens = RU.fromListUnboxed (R.Z R.:. (1::Int) R.:. (numFltrs::Int) R.:. (ftrMpSd::Int) R.:. (ftrMpSd::Int)) b
  convOutpt <- (CV.conv4D input (1:inpImgDim) (R.delay w) (numFltrs:fltrDim) strd ftrMpSd)
  let thresh = (0.0 :: Double)
      actvtn =  (R.+^) convOutpt b_tens
      abvThresh = R.map (\e -> if e > thresh then (e - thresh) else 0) actvtn
      outP = R.reshape outPShape abvThresh
  return outP

cnctdLyr3 input = do
  -- input has extent 1, 32 * 9 * 9
  let asrt = (assert_eq (R.extent input) (R.Z R.:. (1::Int) R.:. (2592::Int)) "ly3 input") `strct` input `debug` "lyr3 inp assrt"
      nIn = (32 * 9 * 9 :: Int) `strct` asrt  -- Number of inputs
      nOut = 256 :: Int -- Number of neurons
      wBnd = sqrt (6.0 / (fromIntegral (nIn + nOut)))
      w = RR.randomishDoubleArray (R.Z R.:. nIn R.:. nOut) (-wBnd) wBnd 1
      b = R.fromUnboxed (R.Z R.:. (1 :: Int) R.:. nOut) (VUN.replicate nOut (0 :: Double))
      thresh = 0.0
  inputC <- R.computeUnboxedP input
  let actvtn = (R.+^) (RM.mmultS inputC w) b
      abvThresh = R.map (\e -> if e > thresh then (e - thresh) else 0) actvtn
      outP = abvThresh
      -- Validate outP has extent 256
      asrt_o = (assert_eq (R.extent outP) (R.Z R.:. (1::Int) R.:. (256::Int)) "ly3 outP") `debug` "asrt_o"
  return (outP `strct` asrt_o `debug` ("lyr3 out assrt" ++ show(asrt_o)))

outptLyr4 input= do
  -- input has extent 256
  let asrt = (assert_eq (R.extent input) (R.Z R.:. (1::Int) R.:. (256::Int)) "ly4 input") `strct` input `debug` "lyr4 inp assrt"
      nIn = (256 :: Int) `strct` asrt  -- Number of inputs
      nOut = length CF.availActns :: Int -- Number of neurons
      wBnd = sqrt (6.0 / (fromIntegral (nIn + nOut)))
      w = RR.randomishDoubleArray (R.Z R.:. nIn R.:. nOut) (-wBnd) wBnd 1
      b = R.fromUnboxed (R.Z R.:. (1 :: Int) R.:. nOut) (VUN.replicate nOut (0 :: Double))
  inputC <- R.computeUnboxedP input
  let actvtn = (R.+^) (RM.mmultS inputC w) b
      outP = VUN.fromList (R.toList actvtn)
            -- Validate outP has extent (length availActns)
      asrt_o = (assert_eq (VUN.length outP) (length CF.availActns) "ly4 outP") `debug` "asrt_o"
  return (outP `strct` asrt_o `debug` ("lyr4 out assrt" ++ show(asrt_o)))