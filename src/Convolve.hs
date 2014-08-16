module Convolve where
import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.Repr.Unboxed as RU
import qualified Data.Array.Repa.Index as RI
import qualified Data.Array.Repa.Algorithms.Convolve as RC


wrap x = do 
  (return x)

apndWrpedE :: (Monad m) => m a -> m [a] -> m [a]
apndWrpedE mx macc = do 
    x <- mx
    acc <- macc
    return (x : acc)

-- A list of the form [m e, m e, m e, m e] where m is a Monad, becomes m [e, e, e, e]
unWrapList :: (Monad m) => [m a] -> m [a]
unWrapList ls = do
  foldr apndWrpedE (wrap []) ls

sol :: R.Shape sh => [Int] -> sh
sol = R.shapeOfList

los :: R.Shape sh => sh -> [Int]
los = R.listOfShape

conv4D
  -- :: (Num a, RU.Unbox a)
  -- => 
  :: (Monad m) 
  => RU.Array R.D RI.DIM4 Double
  -> RU.Array R.D RI.DIM4 Double
  -> Int
  -> m(RU.Array R.D RI.DIM4 Double)

conv4D img fltr strd = do
  -- Neural network convolution
  -- both inputs are 4d tensors, second dimension must match

  -- convenice value - equal 1 + (imgRows- fltRows) / strd
  -- Output: Delayed 4D tensor
  let imgDim = reverse $ los (R.extent img)
      fltrDim = reverse $ los (R.extent fltr)
      ftrMpSd = (1 + quot (imgDim!!2 - fltrDim!!2) strd)
      bRange = [0..(imgDim!!0)-1]
      kRange = [0..(fltrDim!!0)-1]
      combRange = [(b,k) | b <- bRange, k <- kRange] 
      mapHelper :: (Monad m) => (Int, Int) -> m(RU.Array RU.U RI.DIM2 Double)
      mapHelper (b,k) = do
        -- Takes the Image batchSize index and the filter batchSize index
        -- returns a 2d matrix as the resul of convolving using stride strd
        -- img[b, i, : , :] with fltr[k, i, :, :] for all i, and summing over i
        let iRange = [0..(imgDim!!1)-1]
            iResultsM = map conv2D [((R.slice img (R.Z R.:. (b :: Int) R.:. (i :: Int) R.:. R.All R.:. R.All)), (R.slice fltr (R.Z R.:. (k :: Int) R.:. (i :: Int) R.:. R.All R.:. R.All)), strd) | i <- iRange] 
        iResults <- unWrapList iResultsM
        sumOfRes <- R.computeUnboxedP (foldl (R.+^) (head iResults) (tail iResults))
        return (sumOfRes)
      res2DAllbkM = map mapHelper combRange
  res2DAllbk <- unWrapList res2DAllbkM
      -- res2DAllbk is a list of 2d matricies, we need to flatten all the lists, join them in the correct order, and then reshape to the corretly dimension 4d tensor
  let fltn e =
        -- Takes a matirx and flattens it to a list
        let dim = product (los (R.extent e))
        in R.reshape (R.Z R.:. dim) e
      res2DFltnd = map fltn res2DAllbk
      -- All of the data for the 4D tensor in a flat 1d array
      tnsr4DDataFlt = foldl (R.append) (head res2DFltnd) (tail res2DFltnd)
  return (R.reshape (sol [ftrMpSd, ftrMpSd, fltrDim!!0, imgDim!!0]) tnsr4DDataFlt)

conv2D
  :: (Monad m)
  => (RU.Array R.D RI.DIM2 Double, RU.Array R.D RI.DIM2 Double, Int)
  -> m (RU.Array R.D RI.DIM2 Double)

conv2D (img, fltr, strd) = do
  -- vanilla 2d convultion with stride strd - very hackish fuction

  -- Strd 2 case 20 by 20 image convovled with 4 by 4 gives 9 by 9
  if strd == 2 then do
    fltrC <- R.computeUnboxedP fltr
    imgC <- R.computeUnboxedP img
    got <- (RC.convolveOutP RC.outClamp fltrC imgC)
    return (R.traverse got (\_ -> sol [9, 9]) (\f (R.Z R.:. i R.:. j) -> f (sol [2 * j + 2, 2 * i + 2])))
  -- Strd 4 case, 84 by 84 image convovled with 8 by 8 gives 20 by 20
  else do
    fltrC <- R.computeUnboxedP fltr
    imgC <- R.computeUnboxedP img
    got <- (RC.convolveOutP RC.outClamp fltrC imgC)
    return (R.traverse got (\_-> sol [20, 20]) (\f (R.Z R.:. i R.:. j) -> f (sol [4 * j + 4, 4 * i + 4])))
