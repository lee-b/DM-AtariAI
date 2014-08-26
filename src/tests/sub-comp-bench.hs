module Main where
import Data.Time
import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.Algorithms.Convolve as RC
import qualified Data.Array.Repa.Repr.Unboxed as RU
import qualified Data.Array.Repa.Index as RI

main = do

    let img = R.delay $ RU.fromListUnboxed (R.Z R.:. (84 :: Int) R.:. (84 :: Int)) ([1..84*84] :: [Float])
    let fltr = R.delay $ RU.fromListUnboxed (R.Z R.:. (8 :: Int) R.:. (8 :: Int)) ([1..8*8] :: [Float])

    start <- getCurrentTime
    res <- repeateTest1 1000 img fltr
    stop <- getCurrentTime
    print $ (diffUTCTime stop start) / 1000

    --let img = R.delay $ RU.fromListUnboxed (R.Z R.:. (20 :: Int) R.:. (20 :: Int)) ([1..20*20] :: [Float])
    --let fltr = R.delay $ RU.fromListUnboxed (R.Z R.:. (4 :: Int) R.:. (4 :: Int)) ([1..4*4] :: [Float])

    --start <- getCurrentTime
    --res <- repeateTest2 1000 img fltr
    --stop <- getCurrentTime
    --print $ diffUTCTime stop start


repeateTest1 counter img fltr = do
    -- Repeat conv2D on 84x84 8x8 1000 times and print time
    res <- conv2D (img, fltr, 4)
    resCompSum <- R.sumAllP res
    putStrLn $ show resCompSum
    if counter == 0 then do
        return 0
    else do
        repeateTest1 (counter - 1) img fltr

repeateTest2 counter img fltr = do
    -- Repeat conv2D on 20x20 4x4 1000 times
    res <- conv2D (img, fltr, 2)
    putStrLn $ show res
    if counter == 0 then do
        return 0
    else do
        repeateTest1 (counter - 1) img fltr


conv2D
  :: (Monad m)
  => (RU.Array R.D RI.DIM2 Float, RU.Array R.D RI.DIM2 Float, Int)
  -> m (RU.Array RU.U RI.DIM2 Float)
conv2D (img, fltr, strd) = do
  -- vanilla 2d convultion with stride strd - very hackish fuction
  
  -- convolve with repa vanilla function, and then drop elements to satisfy stride strd
  
  -- Use two conditions one for stride 4 and one for stride 2 since these are the only two conditions this function will be used for
  -- Strd 2 case 20 by 20 image convovled with 4 by 4 gives 9 by 9
  -- | strd == 2 = let got = (convolveOutP outClamp (R.computeUnboxedS fltr) (R.computeUnboxedS img))
  --              in ((R.traverse got (\_-> (R.Z R.:. (9:: Int) R.:. (9:: Int))) (\f (R.Z R.:. i R.:. j) -> f (R.Z R.:. (2 * i + 2) R.:. (2 * j + 2)))))
  -- Strd 4 case, 84 by 84 image convovled with 8 by 8 gives 20 by 20
  -- | strd == 4 = let got = (convolveOutP outClamp (R.computeUnboxedS fltr) (R.computeUnboxedS img))
  --              in ((R.traverse got (\_-> (R.Z R.:. (20:: Int) R.:. (20:: Int))) (\f (R.Z R.:. i R.:. j) -> f (R.Z R.:. (4 * i + 4) R.:. (4 * j + 4)))))
  -- | otherwise = error ("Stride size nt supported sorry!: stride " ++ show(strd))

  -- | otherwise = error ("Stride size nt supported sorry!: stride " ++ show(strd))
  if strd == 2 then do
    fltrC <- R.computeUnboxedP fltr
    imgC <- R.computeUnboxedP img
    got <- (RC.convolveOutP RC.outClamp fltrC imgC)
    --return ((R.traverse got (\_-> (R.Z R.:. (9:: Int) R.:. (9:: Int))) (\f (R.Z R.:. i R.:. j) -> f (R.Z R.:. (2 * i + 2) R.:. (2 * j + 2)))))
    return got
  else do
    fltrC <- R.computeUnboxedP fltr
    imgC <- R.computeUnboxedP img
    got <- (RC.convolveOutP RC.outClamp fltrC imgC)
    --return ((R.traverse got (\_-> (R.Z R.:. (20:: Int) R.:. (20:: Int))) (\f (R.Z R.:. i R.:. j) -> f (R.Z R.:. (4 * i + 4) R.:. (4 * j + 4)))))
    return got
