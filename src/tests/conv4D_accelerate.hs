-- ghc conv4D_accelerate.hs -O2 -threaded -XTypeOperators
module Main where

import qualified Data.Array.Repa.Algorithms.Randomish as RR
import qualified Data.Array.Accelerate as A
--import qualified Data.Array.Accelerate.Interpreter as BE
import qualified Data.Array.Accelerate.CUDA as BE
import qualified Data.Array.Repa as R
import qualified Control.DeepSeq as DS


main = do
  -- Conv2D
  let img = A.use $ A.fromList (A.Z A.:. (84 :: Int) A.:. (84 :: Int)) [1..84*84]
      fltr = A.use $ A.fromList (A.Z A.:. (8 :: Int) A.:. (8 :: Int)) [1..64]
  res <- conv2D (img,fltr,4)
  putStrLn  $ show (BE.run res)

  -- Conv4D w/ res 1 16 20 20
  let img = A.use $ A.fromList (A.Z A.:. (1 :: Int) A.:. (4 :: Int) A.:. (84 :: Int) A.:. (84 :: Int)) [1..84*84*4]
  let fltr = A.use $ A.fromList (A.Z A.:. (16 :: Int) A.:. (4 :: Int) A.:. (8 :: Int) A.:. (8 :: Int)) [1..16*4*8*8]

  res1 <- conv4DDeprecated img [1, 4, 84, 84] fltr [16, 4, 8, 8] 4 20
  putStrLn  $ show (res1)
  --putStrLn  $ show (BE.run res1)

  -- Conv4D w/ res 1 32 9 9
  let repeat = do
        let img = A.use $ A.fromList (A.Z A.:. (1 :: Int) A.:. (16 :: Int) A.:. (20 :: Int) A.:. (20 :: Int)) [1..16*20*20]
        let fltr = A.use $ A.fromList (A.Z A.:. (32 :: Int) A.:. (16 :: Int) A.:. (4 :: Int) A.:. (4 :: Int)) [1..32*16*4*4]

        res2 <- conv4DDeprecated img [1, 16, 20, 20] fltr [32, 16, 4, 4] 2 9
        --putStrLn  $ show (res2)
        putStrLn $ show (A.indexArray (BE.run res2) (A.Z A.:. (0 :: Int) A.:. (0 :: Int) A.:. (5 :: Int) A.:. (5 :: Int)))
        repeat
  repeat

  return ()


--strct :: DS.NFData a => c -> a -> c
strct = flip seq -- DS.deepseq

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
  return (A.reshape (A.lift (A.Z A.:. (imgDim!!0) A.:. (fltrDim!!0) A.:. ftrMpSd A.:. ftrMpSd)) tnsr4DDataFlt)

conv2D :: (Monad m)
       => (A.Acc (A.Array A.DIM2 Double), 
           A.Acc (A.Array A.DIM2 Double), 
           Int)
       -> m (A.Acc (A.Array A.DIM2 Double))
conv2D (img, fltr, strd) = do
  -- Wraps a convolution function and provides (naive) stride support
  if strd == 2 then do
    --Strd 2 case 20 by 20 image convovled with 4 by 4 gives 9 by 9
    let got = A.stencil (convolveStencil4x4 fltr) A.Clamp img
    let indxs = A.fromList (A.Z A.:. 9) [2,4..18]
    let sliced = layercake2D (A.use indxs) (A.use indxs) got
    --let sliced = A.use $ A.fromList (A.Z A.:. (9 :: Int) A.:. (9 :: Int)) [1.79 | _ <- [1..400]]
    return sliced
  else do
    --Strd 4 case, 84 by 84 image convovled with 8 by 8 gives 20 by 20
    let got = A.stencil (convolveStencil8x8 fltr) A.Clamp img
    let indxs = A.fromList (A.Z A.:. 20) [4,8..80]
    let sliced = layercake2D (A.use indxs) (A.use indxs) got
    --let sliced = A.use $ A.fromList (A.Z A.:. (20 :: Int) A.:. (20 :: Int)) [1.79 | _ <- [1..400]]
    return sliced

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
        --`strct` filter -- `strct` stencil 
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
       -- `strct` filter -- `strct` stencil 
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
      sliced = A.backpermute 
               (A.index2 rows cols)
               (\ix -> let A.Z A.:. j A.:. i = A.unlift ix 
                      in A.index2 (sl A.! A.index1 j) i)
               xs
  in sliced

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

matSum :: (A.IsNum e, A.Elt e, A.Shape sh) => 
       A.Acc (A.Array sh e)
       -> A.Acc (A.Array sh e)
       -> A.Acc (A.Array sh e)
matSum arr brr =
  -- Element wise sum of two n-dimensional matricies
  A.zipWith (+) arr brr