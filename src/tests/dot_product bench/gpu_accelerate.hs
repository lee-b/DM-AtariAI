
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
  stop <- getCurrentTime
  print $ (diffUTCTime stop start) / (P.fromIntegral n_trials)
  putStrLn $ show res


repeateTest counter accum =
    let res = dotp_example 6000 in
    if counter == 0 then
      res
    else
      repeateTest (counter - 1) (accum + res)

dotp :: Acc (Vector Float) -> Acc (Vector Float) -> Acc (Scalar Float)
dotp xs ys =
  A.fold (+) 0 (A.zipWith (*) xs ys)

dotp_example n =
  let xs_arr = A.fromList (A.Z A.:. (n :: Int)) [1.2 | _ <- [1..n]]
      ys_arr = A.fromList (A.Z A.:. (n :: Int)) [1.232 | _ <- [1..n]]
      res = 0.000001 * A.indexArray ((A.run1 (dotp (A.use xs_arr))) ys_arr) (A.Z)
  in res