module Main where
import Data.Time
import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.Algorithms.Convolve as RC
import qualified Data.Array.Repa.Repr.Unboxed as RU
import qualified Data.Array.Repa.Index as RI

main = do

    let img = R.delay $ RU.fromListUnboxed (R.Z R.:. (84 :: Int) R.:. (84 :: Int)) ([1 | _ <- [1..84*84]] :: [Float])
    let fltr = R.delay $ RU.fromListUnboxed (R.Z R.:. (8 :: Int) R.:. (8 :: Int)) ([0..63] :: [Float])
    putStrLn $ show $ fltr R.! (R.Z R.:. (7 :: Int) R.:. (0 :: Int))

    start <- getCurrentTime
    res <- repeateTest 1000 img fltr 4 0
    stop <- getCurrentTime
    print $ (diffUTCTime stop start) / 1000

repeateTest counter img fltr strd acc = do
    res <- conv2D (img, fltr, strd)
    resSum <- R.sumAllP res
    if counter == 0 then do
        return 0
        putStrLn $ show (R.extent res)
        putStrLn $ show acc
    else do
        repeateTest (counter - 1) img fltr strd (acc + resSum)


conv2D
  :: (Monad m)
  => (RU.Array R.D RI.DIM2 Float, RU.Array R.D RI.DIM2 Float, Int)
  -> m (RU.Array RU.U RI.DIM2 Float)
conv2D (img, fltr, strd) = do
  if strd == 2 then do
    fltrC <- R.computeUnboxedP fltr
    imgC <- R.computeUnboxedP img
    got <- (RC.convolveOutP (RC.outAs 0) fltrC imgC)
    --return ((R.traverse got (\_-> (R.Z R.:. (9:: Int) R.:. (9:: Int))) (\f (R.Z R.:. i R.:. j) -> f (R.Z R.:. (2 * i + 2) R.:. (2 * j + 2)))))
    return got
  else do
    fltrC <- R.computeUnboxedP fltr
    imgC <- R.computeUnboxedP img
    got <- (RC.convolveOutP (RC.outAs 0) fltrC imgC)
    --return ((R.traverse got (\_-> (R.Z R.:. (20:: Int) R.:. (20:: Int))) (\f (R.Z R.:. i R.:. j) -> f (R.Z R.:. (4 * i + 4) R.:. (4 * j + 4)))))
    return got
