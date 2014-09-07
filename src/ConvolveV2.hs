-- Padding

{-# OPTIONS_GHC -fcontext-stack=100 -Odph -rtsopts -threaded 
-fno-liberate-case -funfolding-use-threshold1000 
-funfolding-keeness-factor1000 -fllvm -optlo-O3 #-}
{-# LANGUAGE BangPatterns, TypeOperators #-}


module ConvolveV2 where
import Data.List
import Data.List.Split 
import qualified Data.Array.Accelerate.IO as A
import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.Unsafe as R
import qualified Data.Array.Repa.Repr.Unboxed as RU
import qualified Utils as U
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.CUDA as BE
--import qualified Data.Array.Accelerate.Interpreter as BE

flatten !e =
  -- Takes a matirx and flattens it to a list
  let !dim = product (U.los (R.extent e))
  in R.reshape (R.Z R.:. dim) e
{-# INLINE flatten #-}

conv4D
  :: (Monad m) 
  => RU.Array R.D R.DIM4 Float
  -> RU.Array R.D R.DIM4 Float
  -> Int
  -> m(RU.Array R.D R.DIM4 Float)
conv4D !img !fltr !strd = do
      -- 1          x imageDepth  x imageWidth  x imageHeight
  let _:(!imgDpth):(!imgWdth):(!imgHght):_ = R.deepSeqArrays [img, fltr] (reverse $ U.los $ R.extent img)
      -- numFilters x filterDepth x filterWidth x filterHeight
      (!numFltrs):(!fltrDpth):(!fltrWdth):_ = reverse $ U.los $ R.extent fltr
      !imgList = [R.unsafeSlice img (R.Any R.:. (0 :: Int) R.:. i R.:. R.All R.:. R.All) 
                | i <- [0..(imgDpth - 1)]]
      !signalList = take (numFltrs * imgDpth) (cycle imgList)
      !fltrSlcInd = [(a,b) | a <- [0..(numFltrs - 1)], b <- [0..(fltrDpth - 1)]]
      !fltrList = [R.unsafeSlice fltr (R.Any R.:. d1 R.:. d2 R.:. R.All R.:. R.All) 
                  | (d1, d2) <- fltrSlcInd]
      --signalListDummyTest = take (numFltrs * fltrDpth) $ repeat $ R.delay (R.fromListUnboxed (U.sol [imgWdth, imgHght]) [0.123 | _ <-[1..imgWdth*imgHght]])
      --fltrListDummyTest = take (numFltrs * fltrDpth) $ repeat $ R.delay (R.fromListUnboxed (U.sol [fltrWdth, fltrWdth]) [0.123 | _ <-[1..fltrWdth*fltrWdth]])
  !convResults <- conv2D signalList fltrList strd
  let !resChunks = chunksOf imgDpth convResults
      addRepaArrLst !ls = foldl' (R.+^) (head ls) (tail ls)
      !summedRes = map (\e -> flatten (addRepaArrLst e)) resChunks
      !appendedRes = foldl' R.append (head summedRes) (tail summedRes)
      !outWdthHght =  1 + quot (imgWdth - fltrWdth) strd
      !resTensor = R.reshape (U.sol $ reverse [1, numFltrs, outWdthHght, outWdthHght]) 
                              appendedRes
      --resTensorDummyTest = R.delay (R.fromListUnboxed ((U.sol $ reverse [1, numFltrs, outWdthHght, outWdthHght]) :: R.DIM4)
      --                             [0.123 :: Float | _ <-[1..numFltrs*outWdthHght*outWdthHght]])
  return resTensor
--{-# INLINE conv4D #-}


stencil5ToList :: (t, t, t, t, t) -> [t]
stencil5ToList !(e1,e2,e3,e4,e5) = [e1,e2,e3,e4,e5]
{-# INLINE stencil5ToList #-}


stencil9ToList :: (t, t, t, t, t, t, t, t, t) -> [t]
stencil9ToList !(e1,e2,e3,e4,e5,e6,e7,e8,e9) = [e1,e2,e3,e4,e5,e6,e7,e8,e9]
{-# INLINE stencil9ToList #-}

convolveStencil4x4 :: A.Acc (A.Array (((A.Z A.:. Int) A.:. Int) A.:. Int) Float)
                   -> ((A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float),
                       (A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float),
                       (A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float),
                       (A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float),
                       (A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float))
                   -> ((A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int),
                       (A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int),
                       (A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int),
                       (A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int),
                       (A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int))
                   -> A.Exp Float
convolveStencil4x4 !filtersTensor !stencil1 !stencil2 = 
  let !fltrChoice = stencil5ToList ((stencil5ToList stencil2) !! 0) !! 0
      !indList = ([(r,c) | r <- [0..3], c <- [0..3]] :: [(Int, Int)]) 
      indSten (!r,!c) = stencil5ToList ((stencil5ToList stencil1) !! r) !! c
      indFilter (!r,!c) =  filtersTensor A.! (A.lift (A.Z A.:. r A.:. c A.:. fltrChoice))
  in foldl' (\acc ind -> acc + (indSten ind) * (indFilter ind)) 0 indList
--{-# INLINE convolveStencil4x4 #-}

convolveStencil8x8 :: A.Acc (A.Array (((A.Z A.:. Int) A.:. Int) A.:. Int) Float)
                   -> ((A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, 
                      A.Exp Float, A.Exp Float, A.Exp Float),
                       (A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, 
                        A.Exp Float, A.Exp Float, A.Exp Float),
                       (A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, 
                        A.Exp Float, A.Exp Float, A.Exp Float),
                       (A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, 
                        A.Exp Float, A.Exp Float, A.Exp Float),
                       (A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, 
                        A.Exp Float, A.Exp Float, A.Exp Float),
                       (A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, 
                        A.Exp Float, A.Exp Float, A.Exp Float),
                       (A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, 
                        A.Exp Float, A.Exp Float, A.Exp Float),
                       (A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, 
                        A.Exp Float, A.Exp Float, A.Exp Float),
                       (A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, A.Exp Float, 
                        A.Exp Float, A.Exp Float, A.Exp Float))
                   -> ((A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, 
                      A.Exp Int, A.Exp Int, A.Exp Int),
                       (A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, 
                        A.Exp Int, A.Exp Int, A.Exp Int),
                       (A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, 
                        A.Exp Int, A.Exp Int, A.Exp Int),
                       (A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, 
                        A.Exp Int, A.Exp Int, A.Exp Int),
                       (A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, 
                        A.Exp Int, A.Exp Int, A.Exp Int),
                       (A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, 
                        A.Exp Int, A.Exp Int, A.Exp Int),
                       (A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, 
                        A.Exp Int, A.Exp Int, A.Exp Int),
                       (A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, 
                        A.Exp Int, A.Exp Int, A.Exp Int),
                       (A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, 
                        A.Exp Int, A.Exp Int, A.Exp Int))
                   -> A.Exp Float
convolveStencil8x8 !filtersTensor !stencil1 !stencil2 = 
  let !fltrChoice = stencil9ToList ((stencil9ToList stencil2) !! 0) !! 0
      !indList = ([(r,c) | r <- [0..7], c <- [0..7]] :: [(Int, Int)]) 
      indSten (!r,!c) = stencil9ToList ((stencil9ToList stencil1) !! r) !! c
      indFilter (!r,!c) =  filtersTensor A.! (A.lift (A.Z A.:. r A.:. c A.:. fltrChoice))
  in foldl' (\acc ind -> acc + (indSten ind) * (indFilter ind)) 0 indList
--{-# INLINE convolveStencil8x8 #-}


interleave [] !ys =  ys
interleave !(x:xs) !ys = R.deepSeqArray x (x:interleave ys xs)
{-# INLINE interleave #-}

conv2D 
  :: (Monad m)
  => [RU.Array R.D R.DIM2 Float]
  -> [RU.Array R.D R.DIM2 Float]
  -> Int
  -> m([RU.Array R.D R.DIM2 Float])
conv2D !imgs !fltrs !strd = do
  -- convole the nth img with the nth signal using stride of size strd
  -- All imgs must be same size
  -- All filters must be same size
  -- imgs and fltrs must be lists of same length
  let (R.Z R.:. imgDim1 R.:. imgDim2) = R.extent (head imgs)
      (R.Z R.:. fltrDim1 R.:. fltrDim2) = R.extent (head fltrs)
      !padd =  R.fromListUnboxed
                (U.sol [fltrDim2, imgDim1])
                (take (imgDim1 * fltrDim2) $ repeat 0.0)
      !paddings = take (length imgs - 1) $ repeat (R.delay padd)
      !signalList = interleave imgs paddings
      mask !i = R.fromListUnboxed
                (U.sol [imgDim2 + fltrDim2, imgDim1])
                (take (imgDim1 * (imgDim2 + fltrDim2)) $ repeat (i :: Int))
      !maskList = [R.delay (mask i) | i <- [0..(length imgs -1)]]
  !signalArray <- (A.computeAccP (foldl' R.append 
                                       (head signalList)
                                       (tail signalList)))
  --let signalArray = (A.computeAccS (foldl' R.append 
  --                                     (head signalList)
  --                                     (tail signalList)))
  !signalFltrInd <- (A.computeAccP (foldl' R.append 
                                         (head maskList)
                                         (tail maskList)))
  --let signalFltrInd = (A.computeAccS (foldl' R.append 
  --                                       (head maskList)
  --                                       (tail maskList)))
  let !fltrsReshaped = map (R.reshape ((U.sol [1, fltrDim2, fltrDim1]) :: R.DIM3)) fltrs
  !fltrsTensor <- A.computeAccP (foldl' R.append 
                                      (head fltrsReshaped) 
                                      (tail fltrsReshaped))
  --let fltrsTensor = A.computeAccS (foldl' R.append 
  --                                    (head fltrsReshaped) 
  --                                    (tail fltrsReshaped))
  let !fltrsTensorGPU = (A.use (A.fromRepa fltrsTensor))
      -- Hackish implementation only to be used with filters of size 4 x 4 and size 8 x 8
      !res = if fltrDim1 == 4 then
              R.delay $ A.toRepa $ BE.run (A.stencil2 (convolveStencil4x4 fltrsTensorGPU) -- (A.use (A.fromRepa fltr))) 
                                                          (A.Constant (0.0 :: Float)) 
                                                          (A.use (A.fromRepa signalArray))
                                                          (A.Constant (0 :: Int))
                                                          (A.use (A.fromRepa signalFltrInd)))
            else
              R.delay $ A.toRepa $ BE.run (A.stencil2 (convolveStencil8x8 fltrsTensorGPU) -- (A.use (A.fromRepa fltr))) 
                                                          (A.Constant (0.0 :: Float)) 
                                                          (A.use (A.fromRepa signalArray))
                                                          (A.Constant (0 :: Int))
                                                          (A.use (A.fromRepa signalFltrInd)))
      !startingIndxs = [U.sol [s, 0] | s <- [0,(imgDim2 + fltrDim1)..(length imgs - 1)*(imgDim2 + fltrDim1)]]
      extractFormated !size !arr !strt = R.extract strt size arr
      !extractHelper = extractFormated (U.sol [imgDim2, imgDim1]) res
      !resSegmented = map extractHelper startingIndxs
      applyStrd !array = if strd == 2 then
                          R.unsafeTraverse array (\_ -> U.sol [9, 9])
                                  (\f (R.Z R.:. i R.:. j) -> f (U.sol [2 * j + 2, 2 * i + 2]))
                        else
                          R.unsafeTraverse array (\_-> U.sol [20, 20])
                                         (\f (R.Z R.:. i R.:. j) -> f (U.sol [4 * j + 4, 4 * i + 4]))
      !resSegStrdApplied = map applyStrd resSegmented
      !dummyTestRes = if strd == 2 then
                        take (length imgs) $ repeat (R.delay $ R.fromListUnboxed ((U.sol [9, 9]) :: R.DIM2) [(1.3 :: Float) | _ <- [1..81]])
                      else
                        take (length imgs) $ repeat (R.delay $ R.fromListUnboxed ((U.sol [20, 20]) :: R.DIM2) [(1.43 :: Float) | _ <- [1..400]])
                          
  return resSegStrdApplied
--{-# INLINE conv2D #-}
-- Padding