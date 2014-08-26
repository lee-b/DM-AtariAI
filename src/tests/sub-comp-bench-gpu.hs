module Main where
import Data.Time    
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.CUDA as BE

main = do
    let img = A.use $ A.fromList (A.Z A.:. (84 :: Int) A.:. (84 :: Int)) [1..84*84]
    let fltr = A.use $ A.fromList (A.Z A.:. (8 :: Int) A.:. (8 :: Int)) [1..64]

    start <- getCurrentTime
    res <- repeateTest1 1000 img fltr
    stop <- getCurrentTime
    print $ diffUTCTime stop start


repeateTest1 counter img fltr = do
    -- Repeat conv2D on 84x84 8x8 1000 times and print time
    res <- conv2D (img, fltr, 4)
    let resComp = BE.run (A.sum res)
    putStrLn $ show resComp
    if counter == 0 then do
        return 0
    else do
        repeateTest1 (counter - 1) img fltr


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
    --let indxs = A.fromList (A.Z A.:. 20) [4,8..80]
    --let sliced = layercake2D (A.use indxs) (A.use indxs) got
    --let sliced = A.use $ A.fromList (A.Z A.:. (20 :: Int) A.:. (20 :: Int)) [1.79 | _ <- [1..400]]
    return got

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
  let ((e11,e12,e13,e14,e15,e16,e17,e18,e19),
       (e21,e22,e23,e24,e25,e26,e27,e28,e29),
       (e31,e32,e33,e34,e35,e36,e37,e38,e39),
       (e41,e42,e43,e44,e45,e46,e47,e48,e49),
       (e51,e52,e53,e54,e55,e56,e57,e58,e59),
       (e61,e62,e63,e64,e65,e66,e67,e68,e69),
       (e71,e72,e73,e74,e75,e76,e77,e78,e79),
       (e81,e82,e83,e84,e85,e86,e87,e88,e89),
       (e91,e92,e93,e94,e95,e96,e97,e98,e99)) = stencil
      indList = ([(r,c) | r <- [0..7], c <- [0..7]] :: [(Int, Int)]) 
      indSten (r,c) = stencil9ToList ((stencil9ToList stencil) !! r) !! c
      indFilter (r,c) = filter A.! (A.lift (A.Z A.:. r A.:. c))
  --in foldl (\acc ind -> acc + (indSten ind) * (indFilter ind)) 0 indList
  in (e11 * filter A.! (A.lift (A.Z A.:. (0 :: Int) A.:. (0 :: Int))) +
      e12 * filter A.! (A.lift (A.Z A.:. (0 :: Int) A.:. (1 :: Int))) +
      e13 * filter A.! (A.lift (A.Z A.:. (0 :: Int) A.:. (2 :: Int))) +
      e14 * filter A.! (A.lift (A.Z A.:. (0 :: Int) A.:. (3 :: Int))) +
      e15 * filter A.! (A.lift (A.Z A.:. (0 :: Int) A.:. (4 :: Int))) +
      e16 * filter A.! (A.lift (A.Z A.:. (0 :: Int) A.:. (5 :: Int))) +
      e17 * filter A.! (A.lift (A.Z A.:. (0 :: Int) A.:. (6 :: Int))) +
      e18 * filter A.! (A.lift (A.Z A.:. (0 :: Int) A.:. (7 :: Int))) +
      e21 * filter A.! (A.lift (A.Z A.:. (1 :: Int) A.:. (0 :: Int))) +
      e22 * filter A.! (A.lift (A.Z A.:. (1 :: Int) A.:. (1 :: Int))) +
      e23 * filter A.! (A.lift (A.Z A.:. (1 :: Int) A.:. (2 :: Int))) +
      e24 * filter A.! (A.lift (A.Z A.:. (1 :: Int) A.:. (3 :: Int))) +
      e25 * filter A.! (A.lift (A.Z A.:. (1 :: Int) A.:. (4 :: Int))) +
      e26 * filter A.! (A.lift (A.Z A.:. (1 :: Int) A.:. (5 :: Int))) +
      e27 * filter A.! (A.lift (A.Z A.:. (1 :: Int) A.:. (6 :: Int))) +
      e28 * filter A.! (A.lift (A.Z A.:. (1 :: Int) A.:. (7 :: Int))) +
      e31 * filter A.! (A.lift (A.Z A.:. (2 :: Int) A.:. (0 :: Int))) +
      e32 * filter A.! (A.lift (A.Z A.:. (2 :: Int) A.:. (1 :: Int))) +
      e33 * filter A.! (A.lift (A.Z A.:. (2 :: Int) A.:. (2 :: Int))) +
      e34 * filter A.! (A.lift (A.Z A.:. (2 :: Int) A.:. (3 :: Int))) +
      e35 * filter A.! (A.lift (A.Z A.:. (2 :: Int) A.:. (4 :: Int))) +
      e36 * filter A.! (A.lift (A.Z A.:. (2 :: Int) A.:. (5 :: Int))) +
      e37 * filter A.! (A.lift (A.Z A.:. (2 :: Int) A.:. (6 :: Int))) +
      e38 * filter A.! (A.lift (A.Z A.:. (2 :: Int) A.:. (7 :: Int))) +
      e41 * filter A.! (A.lift (A.Z A.:. (3 :: Int) A.:. (0 :: Int))) +
      e42 * filter A.! (A.lift (A.Z A.:. (3 :: Int) A.:. (1 :: Int))) +
      e43 * filter A.! (A.lift (A.Z A.:. (3 :: Int) A.:. (2 :: Int))) +
      e44 * filter A.! (A.lift (A.Z A.:. (3 :: Int) A.:. (3 :: Int))) +
      e45 * filter A.! (A.lift (A.Z A.:. (3 :: Int) A.:. (4 :: Int))) +
      e46 * filter A.! (A.lift (A.Z A.:. (3 :: Int) A.:. (5 :: Int))) +
      e47 * filter A.! (A.lift (A.Z A.:. (3 :: Int) A.:. (6 :: Int))) +
      e48 * filter A.! (A.lift (A.Z A.:. (3 :: Int) A.:. (7 :: Int))) +
      e51 * filter A.! (A.lift (A.Z A.:. (4 :: Int) A.:. (0 :: Int))) +
      e52 * filter A.! (A.lift (A.Z A.:. (4 :: Int) A.:. (1 :: Int))) +
      e53 * filter A.! (A.lift (A.Z A.:. (4 :: Int) A.:. (2 :: Int))) +
      e54 * filter A.! (A.lift (A.Z A.:. (4 :: Int) A.:. (3 :: Int))) +
      e55 * filter A.! (A.lift (A.Z A.:. (4 :: Int) A.:. (4 :: Int))) +
      e56 * filter A.! (A.lift (A.Z A.:. (4 :: Int) A.:. (5 :: Int))) +
      e57 * filter A.! (A.lift (A.Z A.:. (4 :: Int) A.:. (6 :: Int))) +
      e58 * filter A.! (A.lift (A.Z A.:. (4 :: Int) A.:. (7 :: Int))) +
      e61 * filter A.! (A.lift (A.Z A.:. (5 :: Int) A.:. (0 :: Int))) +
      e62 * filter A.! (A.lift (A.Z A.:. (5 :: Int) A.:. (1 :: Int))) +
      e63 * filter A.! (A.lift (A.Z A.:. (5 :: Int) A.:. (2 :: Int))) +
      e64 * filter A.! (A.lift (A.Z A.:. (5 :: Int) A.:. (3 :: Int))) +
      e65 * filter A.! (A.lift (A.Z A.:. (5 :: Int) A.:. (4 :: Int))) +
      e66 * filter A.! (A.lift (A.Z A.:. (5 :: Int) A.:. (5 :: Int))) +
      e67 * filter A.! (A.lift (A.Z A.:. (5 :: Int) A.:. (6 :: Int))) +
      e68 * filter A.! (A.lift (A.Z A.:. (5 :: Int) A.:. (7 :: Int))) +
      e71 * filter A.! (A.lift (A.Z A.:. (6 :: Int) A.:. (0 :: Int))) +
      e72 * filter A.! (A.lift (A.Z A.:. (6 :: Int) A.:. (1 :: Int))) +
      e73 * filter A.! (A.lift (A.Z A.:. (6 :: Int) A.:. (2 :: Int))) +
      e74 * filter A.! (A.lift (A.Z A.:. (6 :: Int) A.:. (3 :: Int))) +
      e75 * filter A.! (A.lift (A.Z A.:. (6 :: Int) A.:. (4 :: Int))) +
      e76 * filter A.! (A.lift (A.Z A.:. (6 :: Int) A.:. (5 :: Int))) +
      e77 * filter A.! (A.lift (A.Z A.:. (6 :: Int) A.:. (6 :: Int))) +
      e78 * filter A.! (A.lift (A.Z A.:. (6 :: Int) A.:. (7 :: Int))) +
      e81 * filter A.! (A.lift (A.Z A.:. (7 :: Int) A.:. (0 :: Int))) +
      e82 * filter A.! (A.lift (A.Z A.:. (7 :: Int) A.:. (1 :: Int))) +
      e83 * filter A.! (A.lift (A.Z A.:. (7 :: Int) A.:. (2 :: Int))) +
      e84 * filter A.! (A.lift (A.Z A.:. (7 :: Int) A.:. (3 :: Int))) +
      e85 * filter A.! (A.lift (A.Z A.:. (7 :: Int) A.:. (4 :: Int))) +
      e86 * filter A.! (A.lift (A.Z A.:. (7 :: Int) A.:. (5 :: Int))) +
      e87 * filter A.! (A.lift (A.Z A.:. (7 :: Int) A.:. (6 :: Int))) +
      e88 * filter A.! (A.lift (A.Z A.:. (7 :: Int) A.:. (7 :: Int))))
  --in e11
  --in foldl (\acc ind -> acc + (7) * (7)) 0 indList

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
