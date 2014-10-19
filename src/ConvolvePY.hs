{-# OPTIONS_GHC -fcontext-stack=100 #-}
{-# LANGUAGE BangPatterns, TypeOperators #-}

module ConvolvePY where

import qualified GHC.IO.Handle.Types                            as T
import qualified Data.Array.Repa                                as R
import qualified Data.Array.Repa.Unsafe                         as R
import qualified Data.Array.Repa.Repr.Unboxed                   as RU
import qualified Data.ByteString.Char8                          as C
import qualified Configure                                      as CF
import qualified Utils                                          as U

conv4D
  :: T.Handle
  -> T.Handle
  -> RU.Array R.D R.DIM4 Float
  -> RU.Array R.D R.DIM4 Float
  -> Int
  -> IO (RU.Array R.D R.DIM4 Float)
conv4D toC fromC !img !fltr !strd = do
  -- Send data
  C.hPut toC (C.pack (show (R.toList img) ++ ":" ++ (show (R.toList fltr)) ++ ":" ++ (show strd) ++ ":" ++ "\n"))

  -- Get response
  strC <- C.hGetLine fromC
  let str = C.unpack strC 
  --putStrLn ("Haskell got: " ++ take 10 str)
  let ls = (read str) :: [Float]

  if strd == 2 then do
    return (R.delay $ RU.fromListUnboxed (U.sol [9, 9, 32, 1]) ls)
  else do
    return (R.delay $ RU.fromListUnboxed (U.sol [20, 20, 16, 1]) ls)
