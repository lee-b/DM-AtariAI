-- Author: Hamzeh Alsalhi <93hamsal@gmail.com>

module Main where

import Control.DeepSeq as DS
-- import Data.Array.Accelerate.CUDA as BE
import Data.Array.Accelerate.Interpreter as BE
import Data.Array.Repa.Algorithms.Randomish as RR
import Data.List.Split
import Debug.Trace
import System.Directory
import System.Posix.Files
import System.Posix.IO
import System.Process
import System.Random

import qualified Data.Array.Accelerate as A
import qualified Data.Array.Repa as R
import qualified Data.ByteString.Char8 as C
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as VUN

memSz = 10000
aleScrnSz = (160, 210)

debug = flip trace -- var `debug` "Debug Message"
strct = flip seq -- DS.deepseq

assert_eq :: (Eq a, Show a) => a -> a -> [Char] -> Bool
assert_eq x y mrkrMsg =
  let c = if x == y then True  else (error (mrkrMsg ++ show x ++ show y ++ "are not equal!"))
  in c `strct` c

wrap x = do (return x)

apndWrpedE :: (Monad m) => m a -> m [a] -> m [a]
apndWrpedE mx macc = do 
    x <- mx
    acc <- macc
    return (x : acc)

unWrapList :: (Monad m) => [m a] -> m [a]
unWrapList ls = 
  -- A list of the form [m e] where m is a Monad, becomes m [e]
  do foldr apndWrpedE (wrap []) ls

main = 
 -- ## ALE INTERFACE init
 do fo_ex <- doesFileExist "ale_fifo_out"
    fi_ex <- doesFileExist "ale_fifo_in"

    case fo_ex of True -> (readProcess "rm" ["ale_fifo_out"] "")
                  False -> (readProcess "ls" [] "")

    case fi_ex of True -> (readProcess "rm" ["ale_fifo_in"] "")
                  False -> (readProcess "ls" [] "")

    createNamedPipe "ale_fifo_out" $ unionFileModes ownerReadMode 
      ownerWriteMode
    createNamedPipe "ale_fifo_in"  $ unionFileModes ownerReadMode 
      ownerWriteMode

    let toA_IO = openFd "ale_fifo_in"   WriteOnly Nothing 
                  defaultFileFlags
    let fromA_IO = openFd "ale_fifo_out" ReadOnly  Nothing 
                  defaultFileFlags  

    fromA <- fromA_IO
    hndl_in <- fdToHandle fromA

    toA <- toA_IO

    str <- C.hGetLine hndl_in
    --putStrLn str
    fdWrite toA "1,0,0,1\n"

    str <- C.hGetLine hndl_in
    --putStrLn str
    -- first action must be 1 to satrt game
    fdWrite toA "1,18\n"

    foreverPrint hndl_in toA V.empty 0 0

foreverPrint hndl_in fout mem l i = 
 -- ## MAIN LOOP
 mem `seq` i `seq` l `seq` 
 do str <- C.hGetLine hndl_in
    --if (i > 1) then putStrLn (show (VU.foldl (+) 0.0 (V.foldl (VU.++) VU.empty mem))) else putStrLn "Mem too small"
    --let h = map (map (\x -> x + 1.0)) mem
    let strTkns = C.split ':' str
    let (scrStr, epInfo) = (strTkns!!0, strTkns!!1)
    act <- (chooseAction mem i) `strct` strTkns
    fdWrite fout (act ++ ",18\n")
    let smallScr = scrnToNnInp scrStr
    --let smallScr = VUN.fromList [1,2,3]
    putStrLn (show i)
    putStrLn act
    putStrLn ((C.unpack $ C.take 10 scrStr) ++ "... of length: " ++ (show $ C.length scrStr))
    if l >= memSz then
      foreverPrint hndl_in fout (smallScr `V.cons` (V.init mem)) l (i + 1)
    else 
      foreverPrint hndl_in fout (smallScr `V.cons` mem) (l + 1) (i + 1)

scrnToNnInp scr = 
  -- Takes a raw uncroped screen string from ale and makes the full 
  -- transformation into an input ready for the neural network
  -- Param:
    -- scr screen string
  -- Result:
    -- scr screen as list of doubles

  -- Make hex list from string
  let hxLs = chunksOf 2 (C.unpack scr)
  -- Crop 33 rows off of Top and 17 rows off of bottom
      crpHxLs = drop (33 * 160) (take (193 * 160) hxLs)
  -- Downsize to nnImgSz: 80,by 80, XXX: Better Resize
      colDropNnHxLs = map head (chunksOf 2 crpHxLs)
      rowDropNnHxLs = foldl (++) [] (map head (chunksOf 2 (chunksOf 80 colDropNnHxLs)))
  -- XXX Pad with blank pixels from 80,80 to 84,84
      rowDropNnHxLsPded = rowDropNnHxLs ++ ["00" | x <- [1..(84*84 - 80*80)]]
  -- Convert to Grayscale
      grayImg = VUN.fromList [magicArray R.! (R.Z R.:. ((hTD (hex!!1))::Int) R.:. ((hTD  (hex!!0))::Int)) | hex <- rowDropNnHxLsPded]
  in grayImg

availActns = ["0","1","3","4"] -- Actions available to the AI in space invaders

chooseAction mem frmsPlyd = do
    -- Random number generator
    g <- getStdGen
    let epsilon = max ((0.9 :: Float) - ((fromIntegral frmsPlyd) / 1000000.0)) 0.1
    let (rndRl,gN) = randomR (0.0, 1.0) g
    g <- newStdGen
    let (rndIdx,gN)  = randomR (0, (length availActns) - 1) g
    g <- newStdGen
    bestAct <- nnBestAction mem
    if epsilon < rndRl 
      then return (availActns!!rndIdx) 
      else return bestAct

nn = [] -- The neural network

nnBestAction :: (Monad m)
             => V.Vector (VUN.Vector Double)
             -> m ([Char])
nnBestAction mem = do
  -- Main neural netowrk linking layers will be implemented here
  if V.length mem < 4 then 
      return "0"
  else do
      let rcnt4 = (V.take 4 mem) `strct` mem
       -- Stitch last 4 images together into 4D tensor
      let rcnt = VUN.toList (V.foldl (VUN.++) (V.head rcnt4) (V.tail rcnt4)) `strct` rcnt4
      let tens = A.use (A.fromList (A.Z A.:. (1 :: Int) A.:. (4 :: Int) A.:. (84 :: Int) A.:. (84 :: Int)) rcnt) `strct` rcnt
            -- Send input to first layer and propogate through to output layer
      l1O <- cnvLyr1 tens
      l2O <- cnvLyr2 l1O
      l3O <- cnctdLyr3 l2O
      actnProb <- outptLyr4 l3O
        -- Get the most probable action
      return (availActns!!(VUN.maxIndex actnProb) `strct` actnProb)

cnvLyr1 input = do
  -- input has extent 1, 4, 84, 84
  -- let asrt = (assert_eq (R.extent input) (R.Z R.:. (1::Int) R.:. (4::Int) R.:. (84::Int) R.:. (84::Int)) "ly1 input") `strct` input `debug` "lyr1 inp assrt"
  let inpImgDim = [4 :: Int, 84 :: Int, 84 :: Int] `strct` input -- `strct` asrt
      numFltrs = (16 :: Int)
      fltrDim = [4 :: Int, 8 :: Int, 8 :: Int]
      strd = (4 :: Int)
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
      ls_w = R.toList w
      a_w = A.use $ A.fromList (A.Z A.:. (numFltrs::Int) A.:. ((fltrDim!!0)::Int) A.:. ((fltrDim!!1)::Int) A.:. ((fltrDim!!2)::Int)) ls_w
      -- XXX on Neural Netowkr update b should be a list of numFltrs value each replicated ftrMapSd * ftrMpSd times
      b = [0 | _ <- [1..numFltrs * ftrMpSd * ftrMpSd]]
      b_tens = A.use $ A.fromList (A.Z A.:. (1::Int) A.:. (numFltrs::Int) A.:. (ftrMpSd::Int) A.:. (ftrMpSd::Int)) b
  convOutpt <- (conv4DDeprecated input (1:inpImgDim) a_w (numFltrs:fltrDim) strd ftrMpSd)
  -- let asrt_co = (assert_eq (R.extent convOutpt) (R.Z R.:. (1::Int) R.:. (16::Int) R.:. (20::Int) R.:. (20::Int)) "ly1 convOutpt")
  let thresh = A.constant (0.0 :: Double) -- `strct` asrt_co
      actvtn =  matSum convOutpt b_tens
      -- asrt_actv = (assert_eq (R.extent actvtn) (R.Z R.:. (1::Int) R.:. (16::Int) R.:. (20::Int) R.:. (20::Int)) "ly1 actvtn")
      abvThresh = (A.map (\e -> max (e - thresh) 0) actvtn) -- `strct` asrt_actv
      outP = abvThresh
      -- : Validate extent outP is (1, 16, 20, 20)
      -- asrt_o = (assert_eq (R.extent outP) (R.Z R.:. (1::Int) R.:. (16::Int) R.:. (20::Int) R.:. (20::Int)) "ly1 outP")
  return (A.use (BE.run outP))  -- `strct` asrt_o `debug` ("lyr1 out assrt" ++ show(asrt_o)))

cnvLyr2 input = do
  -- input has extent 1, 16, 20 ,20
  -- let asrt = (assert_eq (R.extent input) (R.Z R.:. (1::Int) R.:. (16::Int) R.:. (20::Int) R.:. (20::Int)) "ly2 input") `strct` input `debug` "lyr2 inp assrt"
  let inpImgDim = [16 :: Int, 20 :: Int, 20 :: Int] `strct` input -- `strct` asrt `debug` ("assrt" ++ show(asrt))
      numFltrs = (32 :: Int)
      fltrDim = [16 :: Int, 4 :: Int, 4 :: Int]
      strd = 2 :: Int
      -- number of input connections per neuron
      numInpPer =  (fltrDim!!0 * fltrDim!!1 * fltrDim!!2) :: Int
      -- Feature map dimensions
      ftrMpSd = ((1 + quot (inpImgDim!!1 - fltrDim!!1) strd) :: Int)
      ftrMpDim = [ftrMpSd, ftrMpSd]
      -- number of input connections per neuron
      numOutPer =  (numFltrs * ftrMpDim!!0 * ftrMpDim!!1) :: Int
      wBnd = sqrt (6.0 / (fromIntegral (numInpPer + numOutPer)))
      --XXX enable w random
      w = RR.randomishDoubleArray (R.Z R.:. (numFltrs::Int) R.:. ((fltrDim!!0)::Int) R.:. ((fltrDim!!1)::Int) R.:. ((fltrDim!!2)::Int)) (-wBnd) wBnd 1
      ls_w = R.toList w
      a_w = A.use $ A.fromList (A.Z A.:. (numFltrs::Int) A.:. ((fltrDim!!0)::Int) A.:. ((fltrDim!!1)::Int) A.:. ((fltrDim!!2)::Int)) ls_w
      b = [0 | _ <- [1..numFltrs * ftrMpSd * ftrMpSd]]
      b_tens = A.use $ A.fromList  (A.Z A.:. (1::Int) A.:. (numFltrs::Int) A.:. (ftrMpSd::Int) A.:. (ftrMpSd::Int)) b
  convOutpt <- (conv4DDeprecated input (1:inpImgDim) a_w (numFltrs:fltrDim) strd ftrMpSd)
  let thresh = A.constant (0.0 :: Double)
      actvtn =  matSum convOutpt b_tens
      -- asrt_actv = (assert_eq (R.extent actvtn) (R.Z R.:. (1::Int) R.:. (32::Int) R.:. (9::Int) R.:. (9::Int)) "ly2 actvtn") `debug` "asrt_actv"
      abvThresh = A.map (\e -> max (e - thresh) 0) actvtn `debug` (show $ A.size actvtn) -- `strct` asrt_actv 
      outP = A.reshape (A.lift (A.Z A.:. (1::Int) A.:. (2592::Int))) abvThresh  -- `strct` abvThresh
      -- Validate extent outP is (1, 32 * 9 * 9)
      --asrt_o = (assert_eq (R.extent outP) (R.Z R.:. (1::Int) R.:. (2592::Int)) "ly2 outP") `debug` "asrt_o"
  return (A.use (BE.run outP)) -- `strct` asrt_o `debug` ("lyr2 out assrt" ++ show(asrt_o)))

cnctdLyr3 :: (Monad m) 
          => A.Acc (A.Array A.DIM2 Double) 
          -> m(A.Acc (A.Array A.DIM2 Double))
cnctdLyr3 input = do
  -- input has extent 1, 32 * 9 * 9
  -- let asrt = (assert_eq (R.extent input) (R.Z R.:. (1::Int) R.:. (2592::Int)) "ly3 input") `strct` input `debug` "lyr3 inp assrt"
  let nIn = (32 * 9 * 9 :: Int) `strct` input -- `strct` asrt  -- Number of inputs
      nOut = 256 :: Int -- Number of neurons
      wBnd = sqrt (6.0 / (fromIntegral (nIn + nOut)))
      w = RR.randomishDoubleArray (R.Z R.:. nIn R.:. nOut) (-wBnd) wBnd 1
      ls_w = R.toList w
      a_w = A.use $ A.fromList (A.Z A.:. nIn A.:. nOut) ls_w
      b = A.use $ A.fromList (A.Z A.:. (1 :: Int) A.:. nOut) [0.0 | _ <- [1..nOut]]
      thresh = A.constant 0.0
      actvtn = matSum (matMul input a_w) b
      abvThresh = A.map (\e -> max (e - thresh) 0) actvtn
      outP = abvThresh
      -- Validate outP has extent 256
      -- asrt_o = (assert_eq (R.extent outP) (R.Z R.:. (1::Int) R.:. (256::Int)) "ly3 outP") `debug` "asrt_o"
  return (A.use (BE.run outP))  -- `strct` asrt_o `debug` ("lyr3 out assrt" ++ show(asrt_o)))

outptLyr4 :: (Monad m) 
          => A.Acc (A.Array A.DIM2 Double) 
          -> m(VUN.Vector Double)
outptLyr4 input= do
  -- input has extent 256
  -- let asrt = (assert_eq (R.extent input) (R.Z R.:. (1::Int) R.:. (256::Int)) "ly4 input") `strct` input `debug` "lyr4 inp assrt"
  let nIn = (256 :: Int) `strct` input -- `strct` asrt  -- Number of inputs
      nOut = length availActns :: Int -- Number of neurons
      wBnd = sqrt (6.0 / (fromIntegral (nIn + nOut)))
      w = RR.randomishDoubleArray (R.Z R.:. nIn R.:. nOut) (-wBnd) wBnd 1
      ls_w = R.toList w
      a_w = A.use $ A.fromList (A.Z A.:. nIn A.:. nOut) ls_w
      b = A.use $ A.fromList (A.Z A.:. (1 :: Int) A.:. nOut) [0.0 | _ <- [1..nOut]]
      actvtn = matSum (matMul input a_w) b
      outP = VUN.fromList (A.toList $ BE.run actvtn)
            -- Validate outP has extent (length availActns)
      -- asrt_o = (assert_eq (VUN.length outP) (length availActns) "ly4 outP") `debug` "asrt_o"
  return outP  -- `strct` asrt_o `debug` ("lyr4 out assrt" ++ show(asrt_o)))

conv4D img fltr strd = 
  -- Neural network convolution two 4D tensors, second dimension must match

  -- Paramaters:
  --  img: 4D tensor, The signal to be filtered
  --  fltr: 4D tensor, The kernel to be used on the signal
  --  strd: Number, The size of step the kernel takes when moved  
  -- Output: 4D tensor
  return ()

conv4DDeprecated :: (Monad m) 
                 => A.Acc (A.Array A.DIM4 Double)
                 -> [Int]
                 -> A.Acc (A.Array A.DIM4 Double)
                 -> [Int]
                 -> Int
                 -> Int
                 -> m(A.Acc (A.Array A.DIM4 Double))
conv4DDeprecated img imgDim fltr fltrDim strd ftrMpSd = do
  -- Neural network convolution two 4D tensors, second dimension must match

  -- Paramaters:
  --  img: 4D tensor, The signal to be filtered
  --  imgDim: 4-tuple, (batchSize, numFeatureMaps, numRows, numCols)
  --  fltr: 4D tensor, The kernel to be used on the signal
  --  fltrDim: 4-tuple, (fltBatchSize, numFeatureMaps, numRows, numCols)
  --  convenice value - equal 1 + (imgRows- fltRows) strd
  
  -- Output: Delayed 4D tensor
  let bRange = [0..(imgDim!!0)-1] `strct` img `strct` imgDim`strct` fltr `strct` fltrDim `strct` strd `strct` ftrMpSd
      kRange = [0..(fltrDim!!0)-1]
      combRange = [(b,k) | b <- bRange, k <- kRange] 
      mapHelper :: (Monad m) => (Int, Int) -> m(A.Acc (A.Array A.DIM2 Double))
      mapHelper (b,k) = do
        -- Takes the Image batchSize index and the filter batchSize index
        -- returns a 2d matrix as the resul of convolving using stride strd
        -- img[b, i, : , :] with fltr[k, i, :, :] for all i, and summing over i
        let iRange = [0..(imgDim!!1)-1]
            iResultsM = map conv2D [((A.slice img $ A.lift (A.Z A.:. (b :: Int) A.:. (i :: Int) A.:. A.All A.:. A.All)), (A.slice fltr $ A.lift (A.Z A.:. (k :: Int) A.:. (i :: Int) A.:. A.All A.:. A.All)), strd) | i <- iRange] 
        iResults <- unWrapList iResultsM
        let sumOfRes = foldl (matSum) (head iResults) (tail iResults)
        return (sumOfRes)
      res2DAllbkM = map mapHelper combRange
  res2DAllbk <- unWrapList res2DAllbkM
  -- res2DAllbk is a list of 2d matricies, we need to flatten all the lists, join them in the correct order, and then reshape to the corretly dimension 4d tensor
  let res2DFltnd = map A.flatten res2DAllbk
      -- All of the data for the 4D tensor in a flat 1d array
      tnsr4DDataFlt = foldl (A.++) (head res2DFltnd) (tail res2DFltnd) 
  return (A.reshape (A.lift (A.Z A.:. (imgDim!!0) A.:. (fltrDim!!0) A.:. ftrMpSd A.:. ftrMpSd)) tnsr4DDataFlt) `debug` (show $ A.size tnsr4DDataFlt)

conv2D :: (Monad m)
       => (A.Acc (A.Array A.DIM2 Double), 
           A.Acc (A.Array A.DIM2 Double), 
           Int)
       -> m (A.Acc (A.Array A.DIM2 Double))
conv2D (img, fltr, strd) = do
  -- Wraps a convolution function and provides (naive) stride support
  if strd == 2 then do
    -- Strd 2 case 20 by 20 image convovled with 4 by 4 gives 9 by 9
    let got = A.stencil (convolveStencil4x4 fltr) A.Clamp img `strct` img `strct` fltr 
    let indxs = A.fromList (A.Z A.:. 9) [2,4..18]
    let sliced = layercake2D (A.use indxs) (A.use indxs) got
    return sliced
  else do
    -- Strd 4 case, 84 by 84 image convovled with 8 by 8 gives 20 by 20
    let got = A.stencil (convolveStencil8x8 fltr) A.Clamp img `strct` img `strct` fltr
    let indxs = A.fromList (A.Z A.:. 20) [4,8..80]
    let sliced = layercake2D (A.use indxs) (A.use indxs) got
    return sliced

nnImgSz = (80, 80)

magicArray = 
  R.fromListUnboxed (R.Z R.:. (8::Int) R.:. (16::Int)) magicNumbers
  where magicNumbers = [0.0, 62.56, 51.92, 44.76, 28.56, 31.64, 23.52, 13.440000000000001, 9.520000000000001, 25.72, 37.68, 45.67999999999999, 42.599999999999994, 43.96, 43.32, 42.68, 63.36, 93.12, 77.4, 70.52000000000001, 57.72, 60.239999999999995, 52.959999999999994, 43.44, 39.519999999999996, 55.72, 68.24, 76.24, 74.27999999999999, 78.19999999999999, 74.71999999999998, 73.80000000000001, 106.91999999999999, 123.96, 100.03999999999999, 96.27999999999999, 83.76, 85.71999999999998, 79.28, 70.6, 69.52, 83.16000000000001, 95.68, 106.8, 105.96, 108.75999999999999, 105.0, 104.92, 142.56, 150.83999999999997, 125.52, 119.2, 108.96, 110.36, 104.75999999999999, 97.75999999999999, 95.55999999999999, 109.47999999999999, 122.56, 136.51999999999998, 136.51999999999998, 136.2, 132.44, 132.07999999999998, 174.23999999999998, 173.75999999999996, 144.2, 141.0, 131.04, 132.16, 127.4, 120.12, 118.76, 132.68, 146.04, 160.0, 160.28, 162.79999999999998, 159.04, 155.55999999999997, 198.0, 196.96, 162.88, 159.96, 153.12, 150.84, 146.92, 143.32, 141.12, 152.2, 168.68, 185.76, 186.88000000000002, 186.28, 182.52, 179.04, 217.79999999999998, 219.88, 181.28, 177.8, 174.35999999999999, 171.51999999999998, 168.44, 165.4, 163.20000000000002, 174.56, 191.32000000000002, 205.56, 207.79999999999998, 209.76000000000002, 202.88, 202.24, 233.64000000000001, 239.11999999999998, 199.96, 196.76, 193.32, 190.2, 187.12, 184.92000000000002, 182.71999999999997, 194.07999999999998, 211.12, 228.2, 230.44, 232.39999999999998, 225.51999999999998, 221.76] :: [Double]

hTD h = 
  -- XXX make this function more idomatic
  -- Maps a hex digit to a decimal integer
  case h of '0' -> 0
            '1' -> 1
            '2' -> 2
            '3' -> 3
            '4' -> 4
            '5' -> 5
            '6' -> 6
            '7' -> 7
            '8' -> 8
            '9' -> 9
            'A' -> 10
            'B' -> 11
            'C' -> 12
            'D' -> 13
            'E' -> 14
            'F' -> 15
            _ -> error ("Value error on input: " ++ [h])

stencil5ToList :: (t, t, t, t, t) -> [t]
stencil5ToList (e1,e2,e3,e4,e5) = [e1,e2,e3,e4,e5]

stencil9ToList :: (t, t, t, t, t, t, t, t, t) -> [t]
stencil9ToList (e1,e2,e3,e4,e5,e6,e7,e8,e9) = [e1,e2,e3,e4,e5,e6,e7,e8,e9]

convolveStencil4x4 :: (A.IsNum e, A.Elt e) =>
                   A.Acc (A.Array (A.Plain ((A.Z A.:. Int) A.:. Int)) e)
                   -> ((A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e),
                       (A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e),
                       (A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e),
                       (A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e),
                       (A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e))
                   -> A.Exp e
convolveStencil4x4 filter stencil = 
  let indList = ([(r,c) | r <- [0..3], c <- [0..3]] :: [(Int, Int)]) 
  		  `strct` filter `strct` stencil 
      indSten (r,c) = stencil5ToList ((stencil5ToList stencil) !! r) !! c
      indFilter (r,c) = filter A.! (A.lift (A.Z A.:. r A.:. c))
  in foldl (\acc ind -> acc + (indSten ind) * (indFilter ind)) 0 indList

convolveStencil8x8 :: (A.IsNum e, A.Elt e) =>
                    A.Acc (A.Array (A.Plain ((A.Z A.:. Int) A.:. Int)) e)
                   -> ((A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e, 
                   		A.Exp e, A.Exp e, A.Exp e),
                       (A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e, 
                       	A.Exp e, A.Exp e, A.Exp e),
                       (A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e, 
                       	A.Exp e, A.Exp e, A.Exp e),
                       (A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e, 
                       	A.Exp e, A.Exp e, A.Exp e),
                       (A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e, 
                       	A.Exp e, A.Exp e, A.Exp e),
                       (A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e, 
                       	A.Exp e, A.Exp e, A.Exp e),
                       (A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e, 
                       	A.Exp e, A.Exp e, A.Exp e),
                       (A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e, 
                       	A.Exp e, A.Exp e, A.Exp e),
                       (A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e, A.Exp e, 
                       	A.Exp e, A.Exp e, A.Exp e))
                   -> A.Exp e
convolveStencil8x8 filter stencil = 
  let indList = ([(r,c) | r <- [0..7], c <- [0..7]] :: [(Int, Int)]) 
        `strct` filter `strct` stencil 
      indSten (r,c) = stencil9ToList ((stencil9ToList stencil) !! r) !! c
      indFilter (r,c) = filter A.! (A.lift (A.Z A.:. r A.:. c))
  in foldl (\acc ind -> acc + (indSten ind) * (indFilter ind)) 0 indList

layercake :: A.Elt a =>
          A.Acc (A.Vector Int)
          -> A.Acc (A.Array A.DIM2 a)
          -> A.Acc (A.Array A.DIM2 a)
layercake sl xs =
  -- Slice the rows in list sl from matrix xs
  let A.Z A.:. rows = A.unlift $ A.shape sl
      A.Z A.:. _ A.:. cols = 
        A.unlift $ A.shape xs :: A.Z A.:. A.Exp Int A.:. A.Exp Int
  in A.backpermute 
      (A.index2 rows cols)
      (\ix -> let A.Z A.:. j A.:. i = A.unlift ix 
              in A.index2 (sl A.! A.index1 j) i)
      xs

layercake2D :: A.Elt e =>
            A.Acc (A.Vector Int)
            -> A.Acc (A.Vector Int)
            -> A.Acc (A.Array A.DIM2 e)
            -> A.Acc (A.Array A.DIM2 e)
layercake2D row_sl col_sl xs =
  -- Slice the rows in list row_sl and then columns in col_sl from matrix xs
  let row_sliced = layercake row_sl xs
      row_slicedT = A.transpose row_sliced
      col_slicedT = layercake col_sl row_slicedT
  in A.transpose col_slicedT

matMul :: (A.IsNum e, A.Elt e) => 
          A.Acc (A.Array A.DIM2 e) 
          -> A.Acc (A.Array A.DIM2 e) 
          -> A.Acc (A.Array A.DIM2 e)
matMul arr brr
  -- Accelerate matrix multiplication
  = A.fold (+) 0
  $ A.zipWith (*) arrRepl brrRepl
  where
    A.Z A.:. rowsA A.:. _     = 
      A.unlift (A.shape arr) :: A.Z A.:. A.Exp Int A.:. A.Exp Int
    A.Z A.:. _     A.:. colsB = 
      A.unlift (A.shape brr) :: A.Z A.:. A.Exp Int A.:. A.Exp Int

    arrRepl = A.replicate 
                (A.lift $ A.Z A.:. A.All   A.:. colsB A.:. A.All) 
                arr
    brrRepl = A.replicate 
                (A.lift $ A.Z A.:. rowsA A.:. A.All   A.:. A.All) 
                (A.transpose brr)

matSum :: (A.IsNum e, A.Elt e, A.Shape sh) => 
       A.Acc (A.Array sh e)
       -> A.Acc (A.Array sh e)
       -> A.Acc (A.Array sh e)
matSum arr brr =
  -- Element wise sum of two n-dimensional matricies
  A.zipWith (+) arr brr