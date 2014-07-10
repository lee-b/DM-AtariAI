-- Author: Hamzeh Alsalhi <93hamsal@gmail.com>
-- {-# LANGUAGE BangPatterns, PackageImports #-}
-- {-# OPTIONS -Wall -fno-warn-missing-signatures -fno-warn-incomplete-patterns #-}
module Main where
import Data.List.Split
import System.Directory
import System.Process
import System.Posix.Files
import System.Posix.IO
import System.Random
import Debug.Trace
import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.Shape as RS
import qualified Data.Array.Repa.Algorithms.Randomish as RR
import qualified Data.Array.Repa.Algorithms.Matrix as RM
import qualified Data.Array.Repa.Repr.Unboxed as RU
import qualified Data.Array.Repa.Index as RI
import qualified Data.ByteString.Char8 as C
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as VUN

-- Running List of possible implemetation incorrectness
  -- 4D tensor construvtion from flat list could give wrongly indexed data

memSz = 10000 -- Comes out to ~1.3005 GB of memory 
aleScrnSz = (160, 210)

-- ## Utils
debug = flip trace
-- use: variable `debug` "at variable"
strct = flip seq

assert_eq :: (Eq a, Show a) => a -> a -> [Char] -> Bool
assert_eq x y mrkrMsg =
  let c = if x == y then True  else (error (mrkrMsg ++ show x ++ show y ++ "are not equal!"))
  in c `strct` c
-- ##


-- ## ALE INTERFACE 
main = 
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
-- ##

-- ## MAIN LOOP
foreverPrint hndl_in fout mem l i = 
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


-- Actions available to the AI in the game being played
availActns = ["0","1","3","4"] -- space invaders

chooseAction mem frmsPlyd = do
    -- Random number generator
    g <- getStdGen
    let epsilon = max ((0.9 :: Float) - ((fromIntegral frmsPlyd) / 1000000.0)) 0.1
    let (rndRl,gN) = randomR (0.0, 1.0) g
    g <- newStdGen
    let (rndIdx,gN)  = randomR (0, (length availActns) - 1) g
    g <- newStdGen
    if epsilon < rndRl 
      then return (availActns!!rndIdx) 
      else return (nnBestAction mem)
-- ##

-- ## Neural Net

-- The neural network
nn = []

nnBestAction
  -- :: (Num a, RU.Unbox a)
  -- => 
  :: V.Vector (VUN.Vector Double)
  -> [Char]

nnBestAction mem =
  -- XXX: Main neural netowrk linking layers will be implemented here
      -- Stitch last 4 images together into 4D tensor
  if V.length mem < 4 then 
      "0"
  else
    let rcnt4 = (V.take 4 mem) `strct` mem
        rcnt = (V.foldl (VUN.++) (V.head rcnt4) (V.tail rcnt4)) `strct` rcnt4 
        tens = (R.delay (R.fromUnboxed (R.Z R.:. (1 :: Int) R.:. (4 :: Int) R.:. (84 :: Int) R.:. (84 :: Int)) rcnt)) `strct` rcnt
        -- Send input to first layer and propogate through to output layer
        actnProb = (outptLyr4 (cnctdLyr3 (cnvLyr2 (cnvLyr1 tens)))) `strct` tens
    -- Get the most probable action
    -- XXX activate most prob bease on maxIndex
    -- in availActns!!(VUN.maxIndex actnProb) `strct` actnProb
    in (availActns!!1) `debug` "choice " ++ show (VUN.maxIndex actnProb)
    --in "0" `strct` actnProb -- segfault?
  -- ##


nnBestActionDebug
  -- :: (Num a, RU.Unbox a)
  -- => 
  :: V.Vector (VUN.Vector Double)
  -> [Char]


-- SEGFAULT DEBUGGING
-- Things noticed
-- Sometimes when a layer is ran in isolation it skips its print statements but the print of the frame number in the main loop still works

nnBestActionDebug mem =
  -- A version of the nnBestAction function designed to run each layer in isolation to debug the segfault
  if V.length mem < 4 then 
      "0"
  else
    let rcnt4 = (V.take 4 mem) `strct` mem
        rcnt = (V.foldl (VUN.++) (V.head rcnt4) (V.tail rcnt4)) `strct` rcnt4 
        tens = (R.delay (R.fromUnboxed (R.Z R.:. (1 :: Int) R.:. (4 :: Int) R.:. (84 :: Int) R.:. (84 :: Int)) rcnt)) `strct` rcnt
        -- Send input to first layer and propogate through to output layer

        -- ## SEGFAULT DEBUGGING atleast one of these layers causes a segfault

        -- Lyr 1 res gives us a 4d tensor, to evalutate it in isolation we use depSeqArray tied to the output character literal

        -- SEGFAULT conclusion: YES, Segfaults with isolated use and on tens input
        --lyr1Res = cnvLyr1 $ R.delay (RU.fromListUnboxed (R.Z R.:. (1::Int) R.:. (4::Int) R.:. (84::Int) R.:. (84::Int)) [1..(4 * 84 * 84)])
        -- lyr1Res = cnvLyr1 tens

        -- Lyr 2 res gives us a 2d matrix, to evalutate it in isolation we use depSeqArray tied to the output character literal

        -- SEGFAULT conclusion: YES, Segfaults with isolated use and on the spot constructed input
        --lyr2Res = cnvLyr2 $ R.delay (RU.fromListUnboxed (R.Z R.:. (1::Int) R.:. (16::Int) R.:. (20::Int) R.:. (20::Int)) [1..(16 * 20 * 20)])

        -- Lyr 3 res gives us a 2d matrix, to evalutate it in isolation we use depSeqArray tied to the output character literal

        -- SEGFAULT conclusion: NO, no Segfault with isolated use and on the spot constructed input
        --lyr3Res = cnctdLyr3 $ R.delay (RU.fromListUnboxed (R.Z R.:. (1::Int) R.:. (2592::Int)) [1..2592])

        -- Lyr 4 res gives us an unboxed vector

        -- SEGFAULT conclusion: NO, no Segfault with isolated use and on the spot constructed input
        -- lyr4Res = outptLyr4 $ R.delay (RU.fromListUnboxed (R.Z R.:. (1::Int) R.:. (256::Int)) [1..256])

    -- in (R.deepSeqArray lyr1Res "1") `debug`((show $ R.extent lyr1Res) ++ (show $ R.extent tens)) 
    -- in seq (VUN.length lyr4Res) "1"
    in "1" `debug` show (R.sumAllS tens)

  -- ##

cnvLyr1 input =
  -- XXX input validation
  -- input has extent 1, 4, 84, 84
  let asrt = (assert_eq (R.extent input) (R.Z R.:. (1::Int) R.:. (4::Int) R.:. (84::Int) R.:. (84::Int)) "ly1 input") `strct` input `debug` "lyr1 inp assrt"
      inpImgDim = [4 :: Int, 84 :: Int, 84 :: Int] `strct` asrt
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
      --XXX enable w random
      --w = RR.randomishDoubleArray (R.Z R.:. (numFltrs::Int) R.:. ((fltrDim!!0)::Int) R.:. ((fltrDim!!1)::Int) R.:. ((fltrDim!!2)::Int)) (-wBnd) wBnd 1 
      w =  RU.fromListUnboxed (R.Z R.:. (numFltrs::Int) R.:. ((fltrDim!!0)::Int) R.:. ((fltrDim!!1)::Int) R.:. ((fltrDim!!2)::Int)) [0.01 | _ <- [1..(16 * 8 * 8 * 4)]]
      -- XXX on Neural Netowkr update b should be a list of numFltrs value each replicated ftrMapSd * ftrMpSd times
      b = [0 | _ <- [1..numFltrs * ftrMpSd * ftrMpSd]]
      b_tens = RU.fromListUnboxed (R.Z R.:. (1::Int) R.:. (numFltrs::Int) R.:. (ftrMpSd::Int) R.:. (ftrMpSd::Int)) b
      convOutpt = convolve input (1:inpImgDim) (R.delay w) (numFltrs:fltrDim) strd ftrMpSd 
      asrt_co = (assert_eq (R.extent convOutpt) (R.Z R.:. (1::Int) R.:. (16::Int) R.:. (20::Int) R.:. (20::Int)) "ly1 convOutpt")
      thresh = (0.0 :: Double) `strct` asrt_co
      actvtn =  (R.+^) convOutpt b_tens `debug` "actvtn"
      asrt_actv = (assert_eq (R.extent actvtn) (R.Z R.:. (1::Int) R.:. (16::Int) R.:. (20::Int) R.:. (20::Int)) "ly1 actvtn")
      abvThresh = (R.map (\e -> if e > thresh then (e - thresh) else 0) actvtn) `strct` asrt_actv
      outP = abvThresh
      asrt_o = (assert_eq (R.extent outP) (R.Z R.:. (1::Int) R.:. (16::Int) R.:. (20::Int) R.:. (20::Int)) "ly1 outP") --`debug` "asrt_o"
      -- XXX: Validate extent outP is (1, 16, 20, 20)
  in outP `strct` asrt_o `debug` ("lyr1 out assrt" ++ show(asrt_o))

cnvLyr2 input =
  -- XXX input validation
  -- input has extent 1, 16, 20 ,20

  let asrt = (assert_eq (R.extent input) (R.Z R.:. (1::Int) R.:. (16::Int) R.:. (20::Int) R.:. (20::Int)) "ly2 input") `strct` input `debug` "lyr2 inp assrt"
      inpImgDim = [16 :: Int, 20 :: Int, 20 :: Int] `strct` asrt `debug` ("assrt" ++ show(asrt))
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
      --w = RR.randomishDoubleArray (R.Z R.:. (numFltrs::Int) R.:. ((fltrDim!!0)::Int) R.:. ((fltrDim!!1)::Int) R.:. ((fltrDim!!2)::Int)) (-wBnd) wBnd 1
      w =  RU.fromListUnboxed (R.Z R.:. (numFltrs::Int) R.:. ((fltrDim!!0)::Int) R.:. ((fltrDim!!1)::Int) R.:. ((fltrDim!!2)::Int)) [0.01 | _ <- [1..(32 * 16 * 4 * 4)]]
      b = [0 | _ <- [1..numFltrs * ftrMpSd * ftrMpSd]]
      b_tens = RU.fromListUnboxed  (R.Z R.:. (1::Int) R.:. (numFltrs::Int) R.:. (ftrMpSd::Int) R.:. (ftrMpSd::Int)) b
      convOutpt = convolve input (1:inpImgDim) (R.delay w) (numFltrs:fltrDim) strd ftrMpSd
      thresh = 0.0 :: Double
      actvtn =  (R.+^) convOutpt b_tens
      asrt_actv = (assert_eq (R.extent actvtn) (R.Z R.:. (1::Int) R.:. (32::Int) R.:. (9::Int) R.:. (9::Int)) "ly2 actvtn") `debug` "asrt_actv"
      abvThresh = R.map (\e -> if e > thresh then (e - thresh) else 0) actvtn `strct` asrt_actv 
      outP = R.reshape (R.Z R.:. (1::Int) R.:. (2592::Int)) abvThresh `strct` abvThresh
      asrt_o = (assert_eq (R.extent outP) (R.Z R.:. (1::Int) R.:. (2592::Int)) "ly2 outP") `debug` "asrt_o"
      -- XXX: Validate extent outP is (1, 32 * 9 * 9)
  in outP `strct` asrt_o `debug` ("lyr2 out assrt" ++ show(asrt_o))

cnctdLyr3 input =
  -- XXX input validation
  -- input has extent 1, 32 * 9 * 9

  let asrt = (assert_eq (R.extent input) (R.Z R.:. (1::Int) R.:. (2592::Int)) "ly3 input") `strct` input `debug` "lyr3 inp assrt"
      nIn = (32 * 9 * 9 :: Int) `strct` asrt  -- Number of inputs
      nOut = 256 :: Int -- Number of neurons
      wBnd = sqrt (6.0 / (fromIntegral (nIn + nOut)))
      w = RR.randomishDoubleArray (R.Z R.:. nIn R.:. nOut) (-wBnd) wBnd 1
      -- XXX: Orentation of b could need to be revised col vs row vec
      b = R.fromUnboxed (R.Z R.:. (1 :: Int) R.:. nOut) (VUN.replicate nOut (0 :: Double))
      thresh = 0.0
      actvtn = (R.+^) (RM.mmultS (RU.computeUnboxedS input) w) b
      abvThresh = R.map (\e -> if e > thresh then (e - thresh) else 0) actvtn
      outP = abvThresh
      asrt_o = (assert_eq (R.extent outP) (R.Z R.:. (1::Int) R.:. (256::Int)) "ly3 outP") `debug` "asrt_o"
  -- XXX output validation
  -- output has extent 256
  in outP `strct` asrt_o `debug` ("lyr3 out assrt" ++ show(asrt_o))

outptLyr4 input=
  -- XXX input validation
  -- input has extent 256
  let asrt = (assert_eq (R.extent input) (R.Z R.:. (1::Int) R.:. (256::Int)) "ly4 input") `strct` input `debug` "lyr4 inp assrt"
      nIn = (256 :: Int) `strct` asrt  -- Number of inputs
      nOut = length availActns :: Int -- Number of neurons
      wBnd = sqrt (6.0 / (fromIntegral (nIn + nOut)))
      w = RR.randomishDoubleArray (R.Z R.:. nIn R.:. nOut) (-wBnd) wBnd 1
      b = R.fromUnboxed (R.Z R.:. (1 :: Int) R.:. nOut) (VUN.replicate nOut (0 :: Double))
      actvtn = (R.+^) (RM.mmultS (RU.computeUnboxedS input) w) b
      outP = VUN.fromList (R.toList actvtn)
      asrt_o = (assert_eq (VUN.length outP) (length availActns) "ly4 outP") `debug` "asrt_o"
  -- XXX output validation
  -- output has extent (length availActns)
  in outP `strct` asrt_o `debug` ("lyr4 out assrt" ++ show(asrt_o))

convolve
  -- :: (Num a, RU.Unbox a)
  -- => 
  :: RU.Array R.D RI.DIM4 Double
  -> [Int]
  -> RU.Array R.D RI.DIM4 Double
  -> [Int]
  -> Int
  -> Int
  -> RU.Array R.D RI.DIM4 Double

convolve img imgDim fltr fltrDim strd ftrMpSd = 
  -- Neural network convolution
  -- both inputs are 4d tensors, second dimension must match
  -- Params:
  -- imgDim 4tuple - (batchSize, numFeatureMaps, numRows, numCols)
  -- fltrDim 4tuple - (fltBatchSize, numFeatureMaps, numRows, numCols)
  -- convenice value - equal 1 + (imgRows- fltRows) strd
  -- Output: Delayed 4D tensor
  let bRange = [1..(imgDim!!0)]
      kRange = [1..(fltrDim!!0)]
      combRange = [(b,k) | b <- bRange, k <- kRange] 
      mapHelper :: (Int, Int) -> RU.Array RU.U RI.DIM2 Double
      mapHelper (b,k) = 
        -- Takes the Image batchSize index and the filter batchSize index
        -- returns a 2d matrix as the resul of convolving using stride strd
        -- img[b, i, : , :] with fltr[k, i, :, :] for all i, and summing over i
        let iRange = [1..(imgDim!!1)]
            iResults = map conv2DDebug [((R.slice img (R.Z R.:. (b :: Int) R.:. (i :: Int) R.:. R.All R.:. R.All)), (R.slice fltr (R.Z R.:. (k :: Int) R.:. (i :: Int) R.:. R.All R.:. R.All)), strd) | i <- iRange] 
        in R.computeUnboxedS (foldl (R.+^) (head iResults) (tail iResults)) 
      res2DAllbk = map mapHelper combRange
      -- res2DAllbk is a list of 2d matricies, we need to flatten all the lists, join them in the correct order, and then reshape to the corretly dimension 4d tensor
      fltn e =
        -- Takes a matirx and flattens it to a list
        let dim = product (RS.listOfShape (R.extent e))
        -- in R.delay $ RU.fromListUnboxed (R.Z R.:. dim) ([1.0 | _ <- [1..dim]] :: [Double])-- XXX: remove this false line
        in R.reshape (R.Z R.:. dim) e
      res2DFltnd = map fltn res2DAllbk
      -- All of the data for the 4D tensor in a flat 1d array
      tnsr4DDataFlt = foldl (R.append) (head res2DFltnd) (tail res2DFltnd)
  -- in R.delay $ R.fromListUnboxed (R.Z R.:. (imgDim!!0) R.:. (fltrDim!!0) R.:. ftrMpSd R.:. ftrMpSd) ([1.0 | _ <- [1..(imgDim!!0 * fltrDim!!0 * ftrMpSd * ftrMpSd)]] :: [Double]) XXX: remove this false line
  in R.reshape (R.Z R.:. (imgDim!!0) R.:. (fltrDim!!0) R.:. ftrMpSd R.:. ftrMpSd) tnsr4DDataFlt


conv2D
  -- :: (Num a, RU.Unbox a)
  :: (RU.Array R.D RI.DIM2 Double, RU.Array R.D RI.DIM2 Double, Int)
  -> RU.Array R.D RI.DIM2 Double

conv2D (img, fltr, strd)
  -- vanilla 2d convultion with stride strd - very hackish fuction
  -- XXX: convolve with repa vanilla function, and then drop elements to satisfy stride strd
  -- Use two conditions one for stride 4 and one for stride 2 since these are the only two conditions this function will be used for
  -- Strd 2 case 20 by 20 image convovled with 4 by 4 gives 9 by 9
  | strd == 2 = let got = (convolveOutPDebug outClamp (R.computeUnboxedS fltr) (R.computeUnboxedS img))
                in ((R.traverse got (\_-> (R.Z R.:. (9:: Int) R.:. (9:: Int))) (\f (R.Z R.:. i R.:. j) -> f (R.Z R.:. (2 * i + 2) R.:. (2 * j + 2)))))
  -- Strd 4 case, 84 by 84 image convovled with 8 by 8 gives 20 by 20
  | strd == 4 = let got = (convolveOutPDebug outClamp (R.computeUnboxedS fltr) (R.computeUnboxedS img))
                in ((R.traverse got (\_-> (R.Z R.:. (20:: Int) R.:. (20:: Int))) (\f (R.Z R.:. i R.:. j) -> f (R.Z R.:. (4 * i + 4) R.:. (4 * j + 4)))))
  -- | otherwise = error ("Stride size nt supported sorry!: stride " ++ show(strd))


conv2DDebug
  -- :: (Num a, RU.Unbox a)
  :: (RU.Array R.D RI.DIM2 Double, RU.Array R.D RI.DIM2 Double, Int)
  -> RU.Array R.D RI.DIM2 Double

conv2DDebug (img, fltr, strd)
  -- vanilla 2d convultion with stride strd - very hackish fuction
  -- XXX: convolve with repa vanilla function, and then drop elements to satisfy stride strd
  -- Use two conditions one for stride 4 and one for stride 2 since these are the only two conditions this function will be used for
  -- Strd 2 case 20 by 20 image convovled with 4 by 4 gives 9 by 9
  | strd == 2 = let got  = R.delay $ RU.fromListUnboxed (R.Z R.:. (9::Int) R.:. (9::Int)) ([1..(9*9)] :: [Double])
                    asrt_in_img = (assert_eq (R.extent img) (R.Z R.:. (20::Int) R.:. (20::Int)) "conv2DDebug img")-- `debug` "conv2DDebug asrt_in_img"
                    asrt_in_fltr = (assert_eq (R.extent fltr) (R.Z R.:. (4::Int) R.:. (4::Int)) "conv2DDebug fltr") --`debug` "conv2DDebug asrt_in_fltr"
                --let inp  = R.delay $ RU.fromListUnboxed (R.Z R.:. (20::Int) R.:. (20::Int)) ([1..(20*20)] :: [Double])
                --in R.delay $ (convolveOutPDebug outClamp (RU.fromListUnboxed (R.Z R.:. (9::Int) R.:. (9::Int)) [1..(9*9)]) (RU.fromListUnboxed (R.Z R.:. (20::Int) R.:. (20::Int)) [1..(20*20)])) `R.deepSeqArray` got
                --in (R.computeUnboxedS img) `R.deepSeqArray` got -- SEGFAULT: yes
                --in (img) `R.deepSeqArray` got `debug` show (RU.computeUnboxedS fltr) -- SEGFAULT:
                --in got `debug` ("img sum " ++ show (R.sumAllS img)) -- SEGFAULT: yes
                --in (asrt_in_fltr `seq` asrt_in_img `seq` got) -- `debug` ("fltr sum " ++ show (R.sumAllS fltr)) -- SEGFAULT: no
                --in ((R.traverse inp (\_-> (R.Z R.:. (9:: Int) R.:. (9:: Int))) (\f (R.Z R.:. i R.:. j) -> f (R.Z R.:. (2 * i + 2) R.:. (2 * j + 2))))) -- SEGFAULT: yes
                --in got -- SEGFAULT: no
                --in got `debug` ("conv2DDebug img extent: " ++ show (R.extent img)) -- SEGFAULT:
                --in got `debug` ("conv2DDebug fltr arbitrary index: " ++ show (fltr R.! (R.Z R.:. (0::Int) R.:. (0::Int)) )) -- SEGFAULT: np
                in got `debug` ("conv2DDebug img arbitrary index: " ++ show (img R.! (R.Z R.:. (0::Int) R.:. (0::Int)) )) -- SEGFAULT: Yes
  -- Strd 4 case, 84 by 84 image convovled with 8 by 8 gives 20 by 20
  | strd == 4 = let got  = R.delay $ RU.fromListUnboxed (R.Z R.:. (20::Int) R.:. (20::Int)) ([1..(20*20)] :: [Double])
                    asrt_in_img = (assert_eq (R.extent img) (R.Z R.:. (84::Int) R.:. (84::Int)) "conv2DDebug img") --`debug` "conv2DDebug asrt_in_img"
                    asrt_in_fltr = (assert_eq (R.extent fltr) (R.Z R.:. (8::Int) R.:. (8::Int)) "conv2DDebug fltr") --`debug` "conv2DDebug asrt_in_fltr"
                --let inp  = R.delay $ RU.fromListUnboxed (R.Z R.:. (84::Int) R.:. (84::Int)) ([1..(84*84)] :: [Double])
                --in R.delay $ (convolveOutPDebug outClamp ((RU.fromListUnboxed (R.Z R.:. (9::Int) R.:. (9::Int)) [1..(9*9)])) ((RU.fromListUnboxed (R.Z R.:. (84::Int) R.:. (84::Int)) [1..(84*84)]))) `R.deepSeqArray` got
                --in (R.computeUnboxedS img) `R.deepSeqArray` got -- SEGFAULT: yes
                --in (img) `R.deepSeqArray` got `debug` show (RU.computeUnboxedS fltr) -- SEGFAULT:
                --in got `debug` ("img sum " ++ show (R.sumAllS img)) -- SEGFAULT: yes
                --in (asrt_in_fltr `seq` asrt_in_img `seq` got) -- `debug` ("fltr sum " ++ show (R.sumAllS fltr)) -- SEGFAULT: no
                --in ((R.traverse inp (\_-> (R.Z R.:. (20:: Int) R.:. (20:: Int))) (\f (R.Z R.:. i R.:. j) -> f (R.Z R.:. (4 * i + 4) R.:. (4 * j + 4))))) -- SEGFAULT: yes
                --in got -- SEGFAULT: no
                --in got `debug` ("conv2DDebug img extent: " ++ show (R.extent img)) -- SEGFAULT:
                --in got `debug` ("conv2DDebug fltr arbitrary index: " ++ show (fltr R.! (R.Z R.:. (0::Int) R.:. (0::Int)) )) -- SEGFAULT: No
                in got `debug` ("conv2DDebug img arbitrary index: " ++ show (img R.! (R.Z R.:. (0::Int) R.:. (0::Int)) )) -- SEGFAULT: Yed
  -- | otherwise = error ("Stride size nt supported sorry!: stride " ++ show(strd))


-- ##PREPROCESSOR
-- Screen space interested in, specified as (upperLeftCoord, lowerRightCoord)
nnImgSz = (80, 80)


magicArray = 
  R.fromListUnboxed (R.Z R.:. (8::Int) R.:. (16::Int)) magicNumbers
  where magicNumbers = [0.0, 62.56, 51.92, 44.76, 28.56, 31.64, 23.52, 13.440000000000001, 9.520000000000001, 25.72, 37.68, 45.67999999999999, 42.599999999999994, 43.96, 43.32, 42.68, 63.36, 93.12, 77.4, 70.52000000000001, 57.72, 60.239999999999995, 52.959999999999994, 43.44, 39.519999999999996, 55.72, 68.24, 76.24, 74.27999999999999, 78.19999999999999, 74.71999999999998, 73.80000000000001, 106.91999999999999, 123.96, 100.03999999999999, 96.27999999999999, 83.76, 85.71999999999998, 79.28, 70.6, 69.52, 83.16000000000001, 95.68, 106.8, 105.96, 108.75999999999999, 105.0, 104.92, 142.56, 150.83999999999997, 125.52, 119.2, 108.96, 110.36, 104.75999999999999, 97.75999999999999, 95.55999999999999, 109.47999999999999, 122.56, 136.51999999999998, 136.51999999999998, 136.2, 132.44, 132.07999999999998, 174.23999999999998, 173.75999999999996, 144.2, 141.0, 131.04, 132.16, 127.4, 120.12, 118.76, 132.68, 146.04, 160.0, 160.28, 162.79999999999998, 159.04, 155.55999999999997, 198.0, 196.96, 162.88, 159.96, 153.12, 150.84, 146.92, 143.32, 141.12, 152.2, 168.68, 185.76, 186.88000000000002, 186.28, 182.52, 179.04, 217.79999999999998, 219.88, 181.28, 177.8, 174.35999999999999, 171.51999999999998, 168.44, 165.4, 163.20000000000002, 174.56, 191.32000000000002, 205.56, 207.79999999999998, 209.76000000000002, 202.88, 202.24, 233.64000000000001, 239.11999999999998, 199.96, 196.76, 193.32, 190.2, 187.12, 184.92000000000002, 182.71999999999997, 194.07999999999998, 211.12, 228.2, 230.44, 232.39999999999998, 225.51999999999998, 221.76] :: [Double]

-- XXX make this function more idomatic
hTD h = 
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
-- ##


-- Convolve Out -----------------------------------------------------------------------------------
-- | A function that gets out of range elements from an image.
type GetOut a
  = (RI.DIM2 -> a)   -- ^ The original get function.
  -> RI.DIM2   -- ^ The shape of the image.
  -> RI.DIM2   -- ^ Index of element we were trying to get.
  -> a


-- | Use the provided value for every out-of-range element.
outAs :: a -> GetOut a
outAs x _ _ _ = x


-- | If the requested element is out of range use
--   the closest one from the real image.
outClamp :: GetOut a
outClamp get (_ R.:. yLen R.:. xLen) (sh R.:. j R.:. i)
 = clampX j i
 where
  clampX y x
    | x < 0 = clampY y 0
    | x >= xLen = clampY y (xLen - 1)
    | otherwise = clampY y x
    
  clampY y x
    | y < 0 = get (sh R.:. 0    R.:. x)
    | y >= yLen = get (sh R.:. (yLen - 1) R.:. x)
    | otherwise = get (sh R.:. y    R.:. x)


---- | Image-kernel convolution, 
----   which takes a function specifying what value to use for out-of-range elements.
convolveOutP
  -- :: (Num a, RU.Unbox a)
  :: GetOut Double   -- ^ How to handle out-of-range elements.
  -> RU.Array RU.U RI.DIM2 Double -- ^ Stencil to use in the convolution.
  -> RU.Array RU.U RI.DIM2 Double -- ^ Input image.
  -> RU.Array RU.U RI.DIM2 Double

convolveOutP getOut kernel image
 = kernel `R.deepSeqArray` image `R.deepSeqArray` 
   R.computeUnboxedS $ R.traverse image id stencil
 where  
        krnSh@(R.Z R.:. krnHeight R.:. krnWidth)  = R.extent kernel        
        imgSh@(R.Z R.:. imgHeight R.:. imgWidth)  = R.extent image

        krnHeight2 = krnHeight `div` 2
        krnWidth2  = krnWidth  `div` 2
        krnSize  = RS.size krnSh

        -- If we're too close to the edge of the input image then
        -- we can't apply the stencil because we don't have enough data.
        borderLeft = krnWidth2
        borderRight  = imgWidth   - krnWidth2  - 1
        borderUp = krnHeight2
        borderDown = imgHeight  - krnHeight2 - 1

        -- The actual stencil function.
        stencil get (_ R.:. j R.:. i)
         = let
              get' ix@(_ R.:. y R.:. x)
               | x < borderLeft = getOut get imgSh ix
               | x > borderRight  = getOut get imgSh ix
               | y < borderUp   = getOut get imgSh ix
               | y > borderDown = getOut get imgSh ix
               | otherwise    = get ix

              ikrnWidth' = i - krnWidth2
              jkrnHeight'  = j - krnHeight2

              integrate count acc
               | count == krnSize   = acc
               | otherwise
               = let  ix@(sh R.:. y R.:. x)  = RS.fromIndex krnSh count
                      ix'      = sh R.:. y + jkrnHeight' R.:. x + ikrnWidth'
                      here     = kernel `R.index` ix * (get' ix')
                 in integrate (count + 1) (acc + here)

           in integrate 0 0


---- | Image-kernel convolution, 
----   which takes a function specifying what value to use for out-of-range elements.
convolveOutPDebug
  -- :: (Num a, RU.Unbox a)
  :: GetOut Double   -- ^ How to handle out-of-range elements.
  -> RU.Array RU.U RI.DIM2 Double -- ^ Stencil to use in the convolution.
  -> RU.Array RU.U RI.DIM2 Double -- ^ Input image.
  -> RU.Array RU.U RI.DIM2 Double

 -- Variant to debug segfault
convolveOutPDebug getOut kernel image
  | R.extent image == (R.Z R.:. (84::Int) R.:. (84::Int))   = RU.fromListUnboxed (R.Z R.:. (84::Int) R.:. (84::Int)) [1..(84*84)]
  | R.extent image == (R.Z R.:. (20::Int) R.:. (20::Int))   = RU.fromListUnboxed (R.Z R.:. (20::Int) R.:. (20::Int)) [1..(20*20)]