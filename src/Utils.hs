module Utils where
import Debug.Trace
import qualified Data.Array.Repa as R
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as VUN

wrap x = do 
  (return x)

apndWrpedE :: (Monad m) => m a -> m [a] -> m [a]
apndWrpedE mx macc = do 
    x <- mx
    acc <- macc
    return (x : acc)

-- A list of the form [m e, ...] where m is a Monad, 
-- becomes m [e, e, e, e]
unWrapList :: (Monad m) => [m a] -> m [a]
unWrapList ls = do
  foldr apndWrpedE (wrap []) ls

debug = flip trace
-- use: variable `debug` "at variable"

strct = flip seq

assert_eq :: (Eq a, Show a) => a -> a -> [Char] -> Bool
assert_eq x y mrkrMsg =
  let c = if x == y 
            then True
            else (error (mrkrMsg ++ show x ++ show y ++ "are not equal!"))
  in c

sol :: R.Shape sh => [Int] -> sh
sol = R.shapeOfList

los :: R.Shape sh => sh -> [Int]
los = R.listOfShape
