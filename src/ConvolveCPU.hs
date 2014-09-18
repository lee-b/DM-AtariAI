module ConvolveCPU where

import qualified Data.Array.Repa                                as R
import qualified Data.Array.Repa.Repr.Unboxed                   as RU
import qualified Data.Array.Repa.Index                          as RI
import qualified Data.Array.Repa.Algorithms.Convolve            as RC
import qualified Utils                                          as U

conv4D
  :: (Monad m) 
  => RU.Array R.D RI.DIM4 Float
  -> RU.Array R.D RI.DIM4 Float
  -> Int
  -> m(RU.Array R.D RI.DIM4 Float)
conv4D img fltr strd = do
  -- Inputs: both array inputs are 4d tensors, second dimension must match,
  --         int strd is a stride for the convultion
  -- Output: Delayed 4D tensor
  let imgDim = reverse $ U.los (R.extent img)
      fltrDim = reverse $ U.los (R.extent fltr)
      ftrMpSd = (1 + quot (imgDim !! 2 - fltrDim !! 2) strd)
      bRange = [0..(imgDim !! 0)-1]
      kRange = [0..(fltrDim !! 0)-1]
      combRange = [(b,k) | b <- bRange, k <- kRange] 
      mapHelper :: (Monad m) => (Int, Int) -> m(RU.Array RU.U RI.DIM2 Float)
      mapHelper (b,k) = do
        -- Takes the Image batchSize index and the filter batchSize index
        -- returns a 2d matrix as the resul of convolving using stride strd
        -- img[b, i, : , :] with fltr[k, i, :, :] for all i, and summing over i
        let iRange = [0..(imgDim !! 1)-1]
            iResultsM = map conv2D [((R.slice img (R.Z R.:. (b :: Int)
                                      R.:. (i :: Int) R.:. R.All R.:. R.All)),
                                     (R.slice fltr (R.Z R.:. (k :: Int) 
                                      R.:. (i :: Int) R.:. R.All R.:. R.All)),
                                     strd) | i <- iRange] 
        iResults <- U.unWrapList iResultsM
        sumOfRes <- R.computeUnboxedP (foldl (R.+^) (head iResults)
                      (tail iResults))
        return (sumOfRes)
      res2DAllbkM = map mapHelper combRange
  res2DAllbk <- U.unWrapList res2DAllbkM
  -- res2DAllbk is a list of 2d matricies, we flatten all the lists, join
  -- them in order, and then reshape to the corretly dimension 4d tensor
  let fltn e =
        -- Takes a matirx and flattens it to a list
        let dim = product (U.los (R.extent e))
        in R.reshape (R.Z R.:. dim) e
      res2DFltnd = map fltn res2DAllbk
      -- All of the data for the 4D tensor in a flat 1d array
      tnsr4DDataFlt = foldl (R.append) (head res2DFltnd) (tail res2DFltnd)
  return (R.reshape (U.sol [ftrMpSd, ftrMpSd, fltrDim !! 0, imgDim !! 0])
          tnsr4DDataFlt)

conv2D
  :: (Monad m)
  => (RU.Array R.D RI.DIM2 Float, RU.Array R.D RI.DIM2 Float, Int)
  -> m (RU.Array R.D RI.DIM2 Float)
conv2D (img, fltr, strd) = do
  -- vanilla 2d convultion with stride strd - very hackish fuction

  -- Strd 2 case 20 by 20 image convovled with 4 by 4 gives 9 by 9
  if strd == 2 then do
    fltrC <- R.computeUnboxedP fltr
    imgC <- R.computeUnboxedP img
    got <- (RC.convolveOutP RC.outClamp fltrC imgC)
    return (R.traverse got (\_ -> U.sol [9, 9])
            (\f (R.Z R.:. i R.:. j) -> f (U.sol [2 * j + 2, 2 * i + 2])))
  -- Strd 4 case, 84 by 84 image convovled with 8 by 8 gives 20 by 20
  else do
    fltrC <- R.computeUnboxedP fltr
    imgC <- R.computeUnboxedP img
    got <- (RC.convolveOutP RC.outClamp fltrC imgC)
    return (R.traverse got (\_-> U.sol [20, 20])
            (\f (R.Z R.:. i R.:. j) -> f (U.sol [4 * j + 4, 4 * i + 4])))
