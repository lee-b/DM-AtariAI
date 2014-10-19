{-# OPTIONS_GHC -fcontext-stack=100 #-}
{-# LANGUAGE BangPatterns, TypeOperators #-}

module ConvolveGPU where

import Data.List
import Data.List.Split 
import qualified Data.Vector                                    as V
import qualified Data.Array.Accelerate.IO                       as A
import qualified Data.Array.Repa                                as R
import qualified Data.Array.Repa.Unsafe                         as R
import qualified Data.Array.Repa.Repr.Unboxed                   as RU
import qualified Utils                                          as U
import qualified Data.Array.Accelerate                          as A
import qualified Data.Array.Accelerate.CUDA                     as BE
-- import qualified Data.Array.Accelerate.Interpreter           as BE
import qualified GHC.IO.Handle.Types                            as T

flatten !e =
  -- Takes a matirx and flattens it to a list
  let !dim = product (U.los (R.extent e))
  in R.reshape (R.Z R.:. dim) e

conv4D
  :: T.Handle
  -> T.Handle
  -> RU.Array R.D R.DIM4 Float
  -> RU.Array R.D R.DIM4 Float
  -> Int
  -> IO (RU.Array R.D R.DIM4 Float)
conv4D toC fromC !img !fltr !strd = do
      -- 1          x imageDepth  x imageWidth  x imageHeight
  let _:(!imgDpth):(!imgWdth):(!imgHght):_ = R.deepSeqArrays [img, fltr]
                                                             (reverse $ U.los 
                                                              $ R.extent img)
      -- numFilters x filterDepth x filterWidth x filterHeight
      (!numFltrs):(!fltrDpth):(!fltrWdth):_ = reverse $ U.los $ R.extent fltr
      !imgList = [R.unsafeSlice img (R.Any R.:. (0 :: Int) R.:. i R.:. R.All
                                     R.:. R.All) 
                | i <- [0..(imgDpth - 1)]]
      !signalVector = V.fromList imgList
      !fltrSlcInd = [(a,b) | a <- [0..(numFltrs - 1)],
                             b <- [0..(fltrDpth - 1)]]
      !fltrList = [R.unsafeSlice fltr (R.Any R.:. d1 R.:. d2 R.:. R.All
                                       R.:. R.All) 
                  | (d1, d2) <- fltrSlcInd]
  !convResults <- conv2D signalVector fltrList strd
  let !resChunks = chunksOf imgDpth convResults
      addRepaArrLst !ls = foldl' (R.+^) (head ls) (tail ls)
      !summedRes = map (\e -> flatten (addRepaArrLst e)) resChunks
      !appendedRes = foldl' R.append (head summedRes) (tail summedRes)
      !outWdthHght =  1 + quot (imgWdth - fltrWdth) strd
      !resTensor = R.reshape (U.sol $ reverse [1, numFltrs, outWdthHght,
                                               outWdthHght]) 
                              appendedRes
  return resTensor

stencil5ToList :: (t, t, t, t, t) -> [t]
stencil5ToList !(e1,e2,e3,e4,e5) = [e1,e2,e3,e4,e5]

stencil9ToList :: (t, t, t, t, t, t, t, t, t) -> [t]
stencil9ToList !(e1,e2,e3,e4,e5,e6,e7,e8,e9) = [e1,e2,e3,e4,e5,e6,e7,e8,e9]

type EF = A.Exp Float
type EI = A.Exp Int

convolveStencil4x4 :: A.Acc (A.Array A.DIM3 Float)
                   -> ((EF, EF, EF, EF, EF),
                       (EF, EF, EF, EF, EF),
                       (EF, EF, EF, EF, EF),
                       (EF, EF, EF, EF, EF),
                       (EF, EF, EF, EF, EF))
                   -> ((EI, EI, EI, EI, EI),
                       (EI, EI, EI, EI, EI),
                       (EI, EI, EI, EI, EI),
                       (EI, EI, EI, EI, EI),
                       (EI, EI, EI, EI, EI))
                   -> EF
convolveStencil4x4 !filtersTensor !stencil1 !stencil2 = 
  let !fltrChoice = stencil5ToList ((stencil5ToList stencil2) !! 0) !! 0
      !indList = ([(r,c) | r <- [0..3], c <- [0..3]] :: [(Int, Int)]) 
      indSten (!r,!c) = stencil5ToList ((stencil5ToList stencil1) !! r) !! c
      indFilter (!r,!c) =  filtersTensor A.! (A.lift (A.Z A.:. r A.:. c
                                              A.:. fltrChoice))
  in foldl' (\acc ind -> acc + (indSten ind) * (indFilter ind)) 0 indList

convolveStencil8x8 :: A.Acc (A.Array A.DIM3 Float)
                   -> ((EF, EF, EF, EF, EF, EF, EF, EF, EF),
                       (EF, EF, EF, EF, EF, EF, EF, EF, EF),
                       (EF, EF, EF, EF, EF, EF, EF, EF, EF),
                       (EF, EF, EF, EF, EF, EF, EF, EF, EF),
                       (EF, EF, EF, EF, EF, EF, EF, EF, EF),
                       (EF, EF, EF, EF, EF, EF, EF, EF, EF),
                       (EF, EF, EF, EF, EF, EF, EF, EF, EF),
                       (EF, EF, EF, EF, EF, EF, EF, EF, EF),
                       (EF, EF, EF, EF, EF, EF, EF, EF, EF))
                   -> ((EI, EI, EI, EI, EI, EI, EI, EI, EI),
                       (EI, EI, EI, EI, EI, EI, EI, EI, EI),
                       (EI, EI, EI, EI, EI, EI, EI, EI, EI),
                       (EI, EI, EI, EI, EI, EI, EI, EI, EI),
                       (EI, EI, EI, EI, EI, EI, EI, EI, EI),
                       (EI, EI, EI, EI, EI, EI, EI, EI, EI),
                       (EI, EI, EI, EI, EI, EI, EI, EI, EI),
                       (EI, EI, EI, EI, EI, EI, EI, EI, EI),
                       (EI, EI, EI, EI, EI, EI, EI, EI, EI))
                   -> EF
convolveStencil8x8 !filtersTensor !stencil1 !stencil2 = 
  let !fltrChoice = stencil9ToList ((stencil9ToList stencil2) !! 0) !! 0
      !indList = ([(r,c) | r <- [0..7], c <- [0..7]] :: [(Int, Int)]) 
      indSten (!r,!c) = stencil9ToList ((stencil9ToList stencil1) !! r) !! c
      indFilter (!r,!c) =  filtersTensor A.! (A.lift (A.Z A.:. r A.:. c
                                              A.:. fltrChoice))
  in foldl' (\acc ind -> acc + (indSten ind) * (indFilter ind)) 0 indList


interleave [] !ys =  ys
interleave !(x:xs) !ys = R.deepSeqArray x (x:interleave ys xs)

maskLyr1 !i = R.fromListUnboxed
          ((U.sol [84 + 8, 84]) :: R.DIM2)
          (take (84 * (84 + 8)) $ repeat (i :: Int))
maskListLyr1 = [R.delay (maskLyr1 i) | i <- [0..63]]

signalFltrIndLyr1 = (A.computeAccS (foldl' R.append 
                                         (head maskListLyr1)
                                         (tail maskListLyr1)))

maskLyr2 !i = R.fromListUnboxed
          ((U.sol [20 + 4, 20]) :: R.DIM2)
          (take (20 * (20 + 4)) $ repeat (i :: Int))
maskListLyr2 = [R.delay (maskLyr2 i) | i <- [0..511]]

signalFltrIndLyr2 = (A.computeAccS (foldl' R.append 
                                         (head maskListLyr2)
                                         (tail maskListLyr2)))

-- no trailing padding included
signalArrayShapeLyr1 = ((U.sol [84 + (84 + 8) * (63), 84]) :: R.DIM2)
signalArrayShapeLyr2 = ((U.sol [20 + (20 + 4) * (511), 20]) :: R.DIM2)

signalArrayShapeUnrepLyr1 = ((U.sol [(84 + 8) * (4), 84]) :: R.DIM2)
signalArrayShapeUnrepLyr2 = ((U.sol [(20 + 4) * (16), 20]) :: R.DIM2)

-- trailing padding included
signalArrayShapeRepLyr1 = (A.Z A.:. (84 :: Int) A.:. ((84 + 8) * (64) :: Int))
signalArrayShapeRepLyr2 = (A.Z A.:. (20 :: Int) A.:. ((20 + 4) * (512) :: Int))

placeHolderArr = R.delay $ R.fromListUnboxed ((U.sol [2]) :: R.DIM1) 
                                             ([1..2] :: [Int])


conv2D 
  :: (Monad m)
  => V.Vector (RU.Array R.D R.DIM2 Float)
  -> [RU.Array R.D R.DIM2 Float]
  -> Int
  -> m([RU.Array R.D R.DIM2 Float])
conv2D !imgs !fltrs !strd = do
  -- Hackish implementation only to be used with filters size 4 x 4 and 8 x 8

  -- convole the nth img with the nth signal using stride of size strd
  -- All imgs must be same size
  -- All filters must be same size
  -- imgs and fltrs must be lists of same length
  let (R.Z R.:. imgDim1 R.:. imgDim2) = R.extent (V.head imgs)
      (R.Z R.:. fltrDim1 R.:. fltrDim2) = R.extent (head fltrs)
      nSgnls = if imgDim1 == 84 then 64 :: Int else 512 :: Int 
      nUniqeFltrs = if imgDim1 == 84 then 16 :: Int else 32 :: Int 
  let signalArrayShape = if imgDim1 == 84 then signalArrayShapeLyr1
                                          else signalArrayShapeLyr2
  let signalArrayShapeUnrep = if imgDim1 == 84 then signalArrayShapeUnrepLyr1
                                               else signalArrayShapeUnrepLyr2
  let signalArrayShapeRep = if imgDim1 == 84 then signalArrayShapeRepLyr1
                                             else signalArrayShapeRepLyr2
      signalArrayUnrep = (R.unsafeTraverse 
                            placeHolderArr
                            (\_-> signalArrayShapeUnrep)
                            (\f (R.Z R.:. i R.:. j) -> 
                              let jMod = mod j (imgDim2 + fltrDim2) in
                                if jMod < imgDim2 then
                                  (imgs V.! (mod (quot j (imgDim2 + fltrDim2)) 
                                            (V.length imgs)))
                                  R.! (U.sol [jMod, i])
                                else
                                  0))
  signalArrayUnrepComp <- A.computeAccP signalArrayUnrep
  let signalArrayUnrepAcc = A.use (A.fromRepa signalArrayUnrepComp)
      signalArrayUnrepSl = A.slice signalArrayUnrepAcc (A.lift (A.Z A.:. A.All
                                                                A.:. A.All))
      signalArrayRep = A.replicate (A.lift (A.Z A.:. A.All A.:. (nUniqeFltrs)
                                    A.:. A.All))
                                   signalArrayUnrepSl
      signalArray = A.reshape (A.lift signalArrayShapeRep) signalArrayRep


  let signalFltrInd = if imgDim1 == 84 then signalFltrIndLyr1
                                       else signalFltrIndLyr2
  let !fltrsReshaped = map (R.reshape ((U.sol [1, fltrDim2, fltrDim1])
                                       :: R.DIM3)) fltrs
  let fltrsTensor = A.computeAccS (foldl' R.append 
                                      (head fltrsReshaped) 
                                      (tail fltrsReshaped))
  let !fltrsTensorGPU = (A.use (A.fromRepa fltrsTensor))
      dRR x = R.delay $ A.toRepa $ BE.run x
      !res = if fltrDim1 == 4 then
              dRR (A.stencil2 (convolveStencil4x4 fltrsTensorGPU)
                              (A.Constant (0.0 :: Float)) 
                              (signalArray)
                              (A.Constant (0 :: Int))
                              (A.use (A.fromRepa signalFltrInd)))
            else
              dRR (A.stencil2 (convolveStencil8x8 fltrsTensorGPU)
                              (A.Constant (0.0 :: Float)) 
                              (signalArray)
                              (A.Constant (0 :: Int))
                              (A.use (A.fromRepa signalFltrInd)))
      !startingIndxs = [U.sol [s, 0] | 
                        s <- [0,(imgDim2 + fltrDim1)..(nSgnls - 1)*(imgDim2 +
                                                                    fltrDim1)]]
      extractFormated !size !arr !strt = R.extract strt size arr
      !extractHelper = extractFormated (U.sol [imgDim2, imgDim1]) res
      !resSegmented = map extractHelper startingIndxs
      applyStrd !array = if strd == 2 then
                          R.unsafeTraverse array (\_ -> U.sol [9, 9])
                                  (\f (R.Z R.:. i R.:. j) -> 
                                      f (U.sol [2 * j + 2, 2 * i + 2]))
                        else
                          R.unsafeTraverse array (\_-> U.sol [20, 20])
                                         (\f (R.Z R.:. i R.:. j) ->
                                            f (U.sol [4 * j + 4, 4 * i + 4]))
      !resSegStrdApplied = map applyStrd resSegmented
  return resSegStrdApplied
