
module Main where
import Prelude as P

import Data.Array.Repa as R
import Data.Array.Repa.Eval as R
import Data.Array.Repa.Index as R
import Data.Time

main :: IO ()
main = do
  let n_trials = 1000
  start <- getCurrentTime
  res <- repeateTest n_trials (0.0)
  putStrLn $ show res
  stop <- getCurrentTime
  print $ (diffUTCTime stop start) / (P.fromIntegral n_trials)


repeateTest counter accum = do
    res <- dotp_example 6000
    if counter == 0 then do
      return res
    else do
      repeateTest (counter - 1) (accum + res)

dotp :: (Monad m) => R.Array R.U R.DIM1 Float -> R.Array R.U R.DIM1 Float -> m (Float)
dotp xs ys = do
  dp <- R.foldP (+) 0 (R.zipWith (*) xs ys)
  return $ dp R.! (R.Z)

dotp_example n = do
  let xs_arr = R.fromListUnboxed (R.Z R.:. (n :: Int)) [1.2 | _ <- [1..n]]
      ys_arr = R.fromListUnboxed (R.Z R.:. (n :: Int)) [1.232 | _ <- [1..n]]
  res <- dotp xs_arr ys_arr
  return (0.000001 * res)