{-# LANGUAGE TypeOperators #-}

module Convolve where
import qualified Data.Array.Accelerate.IO as A
import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.Repr.Unboxed as RU
import qualified Utils as U
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.CUDA as BE
--import qualified Data.Array.Accelerate.Interpreter as BE

--conv4D
--  :: (Monad m) 
--  => RU.Array R.D RI.DIM4 Double
--  -> RU.Array R.D RI.DIM4 Double
--  -> Int
--  -> m(RU.Array R.D RI.DIM4 Double)
--conv4D img fltr strd = do
--      -- 1          x imageDepth  x imageWidth  x imageHeight
--  let _:imgDpth:imgWdth:_ = reverse $ U.los $ R.extent img
--      -- numFilters x filterDepth x filterWidth x filterHeight
--      numFltrs:fltrDpth:fltrWdth:_ = reverse $ U.los $ R.extent fltr



-- Step 1
  -- Create array of input signal tensor flattened and repeated n_fltrs times, with gaps of 0 width fltr_w
  -- Create array 2 of same size that indexes filter based on position in the list
  -- Create list of fltrs of size n_fltrs 

-- Step 2
  -- Run the above 2 arrays with a dynamic convolve 2d function that chooses the correct filte for the corret positions
  -- in the convolve 2d function leave the positions skipped over by the stride paramter as 0 to save computations*
    -- *may or may not be worth it since all these computations being in parrallel might not be causing any slowdown


-- Step 3
  -- Drop unecessary rows and columsn from output as specified by stride parameter

-- Step 4
  -- Slpit large output array into groups of arrays to be sumed

-- Step 5
  -- sum each group to get one 2D array per group

-- Step 6
  -- Stitch all of these arrays into a 4D tensor

interleave [] ys = ys
interleave (x:xs) ys = x:interleave ys xs

stencil5ToList :: (t, t, t, t, t) -> [t]
stencil5ToList (e1,e2,e3,e4,e5) = [e1,e2,e3,e4,e5]

convolveStencil4x4 :: A.Acc (A.Array (((A.Z A.:. Int) A.:. Int) A.:. Int) Double)
                   -> ((A.Exp Double, A.Exp Double, A.Exp Double, A.Exp Double, A.Exp Double),
                       (A.Exp Double, A.Exp Double, A.Exp Double, A.Exp Double, A.Exp Double),
                       (A.Exp Double, A.Exp Double, A.Exp Double, A.Exp Double, A.Exp Double),
                       (A.Exp Double, A.Exp Double, A.Exp Double, A.Exp Double, A.Exp Double),
                       (A.Exp Double, A.Exp Double, A.Exp Double, A.Exp Double, A.Exp Double))
                   -> ((A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int),
                       (A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int),
                       (A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int),
                       (A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int),
                       (A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int, A.Exp Int))
                   -> A.Exp Double
convolveStencil4x4 filtersTensor stencil1 stencil2 = 
  let fltrChoice = stencil5ToList ((stencil5ToList stencil2) !! 0) !! 0
      indList = ([(r,c) | r <- [0..3], c <- [0..3]] :: [(Int, Int)]) 
      indSten (r,c) = stencil5ToList ((stencil5ToList stencil1) !! r) !! c
      --indFilter (r,c) = A.fromIntegral fltrChoice 
      --indFilter (r,c) = filtersTensor A.! (A.lift (A.Z A.:. (0 :: Int) A.:. (0 :: Int) A.:. (0 :: Int))) 
      --indFilter (r,c) = A.lift (1.34 :: Double) 
      indFilter (r,c) =  filtersTensor A.! (A.lift (A.Z A.:. r A.:. c A.:. fltrChoice))
  in foldl (\acc ind -> acc + (indSten ind) * (indFilter ind)) 0 indList

conv2D 
  :: (Monad m)
  => [RU.Array R.D R.DIM2 Double]
  -> [RU.Array R.D R.DIM2 Double]
  -> Int
  -> m([A.Array A.DIM2 Double])
conv2D imgs fltrs strd = do
  -- convole the nth img with the nth signal using stride of size strd
  -- All imgs must be same size
  -- All filters must be same size
  -- imgs and fltrs must be lists of same length
  let (R.Z R.:. imgDim1 R.:. imgDim2) = R.extent (head imgs)
      (R.Z R.:. fltrDim1 R.:. fltrDim2) = R.extent (head fltrs)
      padd =  R.fromListUnboxed
                (U.sol [fltrDim2, imgDim1])
                (take (imgDim1 * fltrDim2) $ repeat 0.0)
      paddings = take (length imgs - 1) $ repeat (R.delay padd)
      signalList = interleave imgs paddings
      mask i = R.fromListUnboxed
                (U.sol [imgDim2 + fltrDim2, imgDim1])
                (take (imgDim1 * (imgDim2 + fltrDim2)) $ repeat (i :: Int))
      maskList = [R.delay (mask i) | i <- [0..(length imgs -1)]]
  signalArray <- (A.computeAccP (foldl R.append 
                                       (head signalList)
                                       (tail signalList)))
  signalFltrInd <- (A.computeAccP (foldl R.append 
                                         (head maskList)
                                         (tail maskList)))
  let fltrsReshaped = map (R.reshape ((U.sol [1, fltrDim2, fltrDim1]) :: R.DIM3)) fltrs
  fltrsTensor <- A.computeAccP (foldl R.append 
                                      (head fltrsReshaped) 
                                      (tail fltrsReshaped))
  let fltrsTensorGPU = (A.use (A.fromRepa fltrsTensor))
  let res = BE.run (A.stencil2 (convolveStencil4x4 fltrsTensorGPU) -- (A.use (A.fromRepa fltr))) 
                              A.Clamp 
                              (A.use (A.fromRepa signalArray))
                              A.Clamp 
                              (A.use (A.fromRepa signalFltrInd)))
  return [res]