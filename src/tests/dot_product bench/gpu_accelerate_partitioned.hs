
module Main where
import Prelude as P

import Data.Array.Accelerate            as A
import Data.Array.Accelerate.CUDA       as A
import Data.Time

main :: IO ()
main = do
  let n_trials = 1000
  start <- getCurrentTime
  let res = repeateTest n_trials (0.0)
  putStrLn $ show res
  stop <- getCurrentTime
  print $ (diffUTCTime stop start) / (P.fromIntegral n_trials)


repeateTest counter accum =
    let res = dotp_example 6000000 in
    if counter == 0 then
      res
    else
      repeateTest (counter - 1) (accum + res)

dotp :: Acc (Vector Float) -> Acc (Vector Float) -> Acc (Scalar Float)
dotp xs ys =
  A.fold (+) 0 (A.zipWith (*) xs ys)

dotp_example n =
  let n_partitions = 1
      make_xs_arr len = A.fromList (A.Z A.:. (len :: Int)) [1.2 | _ <- [1..len]]
      make_ys_arr len = A.fromList (A.Z A.:. (len :: Int)) [1.232 | _ <- [1..len]]
      xs_arrays = P.map make_xs_arr [div n n_partitions | _ <- [1..n_partitions]]
      ys_arrays = P.map make_ys_arr [div n n_partitions | _ <- [1..n_partitions]]
      res (xs_arr, ys_arr) = 0.000001 * A.indexArray ((A.run1 (dotp (A.use xs_arr))) ys_arr) (A.Z)
      results = P.foldl (+) 0 (P.map res (P.zip xs_arrays ys_arrays))
  in results