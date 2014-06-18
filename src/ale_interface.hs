-- Author: Hamzeh Alsalhi <93hamsal@gmail.com>

module Main where
import Control.Concurrent
import Data.List.Split
import System.Directory
import System.IO
import System.Process
import System.Posix.Files
import System.Posix.IO
import System.Random
import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.Algorithms.Randomish as RR
import qualified Data.Array.Repa.Algorithms.Matrix as RM
import qualified Data.Array.Repa.Repr.Unboxed as RU
import qualified Data.Array.Repa.Index as RI
import qualified Data.ByteString as B
import qualified Data.ByteString.Char8 as C
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as VU
import qualified Text.Parsec.Token as PT

memSz = 10000 -- Comes out to ~1.3005 GB of memory 
aleScrnSz = (160, 210)

-- ## ALE INTERFACE 
main = 
 do fo_ex <- doesFileExist "ale_fifo_out"
    fi_ex <- doesFileExist "ale_fifo_in"

    case fo_ex of True -> (readProcess "rm" ["ale_fifo_out"] "")
                  False -> (readProcess "ls" [] "")

    case fi_ex of True -> (readProcess "rm" ["ale_fifo_in"] "")
                  False -> (readProcess "ls" [] "")

    createNamedPipe "ale_fifo_out" $ unionFileModes ownerReadMode 
      ownerWriteMode
    createNamedPipe "ale_fifo_in"  $ unionFileModes ownerReadMode 
      ownerWriteMode

    let toA_IO = openFd "ale_fifo_in"   WriteOnly Nothing 
                  defaultFileFlags
    let fromA_IO = openFd "ale_fifo_out" ReadOnly  Nothing 
                  defaultFileFlags  

    fromA <- fromA_IO
    hndl_in <- fdToHandle fromA

    toA <- toA_IO

    str <- C.hGetLine hndl_in
    --putStrLn str
    fdWrite toA "1,0,0,1\n"

    str <- C.hGetLine hndl_in
    --putStrLn str
    -- first action must be 1 to satrt game
    fdWrite toA "1,18\n"

    foreverPrint hndl_in toA V.empty 0 0
-- ##

-- ## MAIN LOOP
foreverPrint hndl_in fout mem l i = 
 mem `seq` i `seq` l `seq` 
 do str <- C.hGetLine hndl_in
    --if (i > 1) then putStrLn (show (VU.foldl (+) 0.0 (V.foldl (VU.++) VU.empty mem))) else putStrLn "Mem too small"
    --let h = map (map (\x -> x + 1.0)) mem
    let strTkns = C.split ':' str
    let (scrStr, epInfo) = (strTkns!!0, strTkns!!1)
    act <- (chooseAction mem i)
    fdWrite fout (act ++ ",18\n")
    let smallScr = scrnToNnInp scrStr
    putStrLn (show i)
    case (l >= memSz) of  
      True ->  foreverPrint hndl_in fout (smallScr `V.cons` (V.init mem)) l (i + 1)
      False -> foreverPrint hndl_in fout (smallScr `V.cons` mem) (l + 1) (i + 1)

-- Actions available to the AI in the game being played
availActns = ["0","1","3","4"] -- sapce invaders


chooseAction mem frmsPlyd = do
    -- Random number generator
    g <- getStdGen
    let epsilon = max ((0.9 :: Float) - ((fromIntegral frmsPlyd) / 1000000.0)) 0.1
    let (rndRl,gN)  = randomR (0.0, 1.0) g
    g <- newStdGen
    let (rndIdx,gN)  = randomR (0, (length availActns) - 1) g
    g <- newStdGen
    if epsilon < rndRl then return (availActns!!rndIdx) else return (nnBestAction [])
-- ##

-- ## Neural Net

-- The neural network
cnvLyr1 input =
  let inpImgDim = [4 :: Int, 84 :: Int, 84 :: Int]
      numFltrs = (16 :: Int)
      fltrDim = [4 :: Int,8 :: Int,8 :: Int]
      fltrStrd = 4 :: Int
      -- number of input connections per neuron
      numInpPer =  (fltrDim!!0 * fltrDim!!1 * fltrDim!!2) :: Int
      -- Feature map dimensions
      ftrMapSide = 1 + (round ((fromIntegral (inpImgDim!!1 - fltrDim!!1)) / 4.0)) :: Int
      ftrMapDim = [ftrMapSide, ftrMapSide]
      -- number of input connections per neuron
      numOutPer =  (numFltrs * ftrMapDim!!0 * ftrMapDim!!2) :: Int
      wBnd = sqrt (6.0 / (fromIntegral (numInpPer + numOutPer)))
      w = RR.randomishDoubleArray (R.Z R.:. (numFltrs::Int) R.:. ((fltrDim!!0)::Int) R.:. ((fltrDim!!1)::Int) R.:. ((fltrDim!!2)::Int)) (-wBnd) wBnd 1
      biasV = R.fromUnboxed (R.Z R.:. (numFltrs::Int)) (VU.replicate numFltrs (0 :: Float))
      convOutpt = 0.0 -- XXX
  in []


cnvLyr2 = []
cnctdLyr3 = []
outptLyr4 = []

nn = []

nnBestAction mem =
  "0"
-- ##

-- ##PREPROCESSOR
-- Screen space interested in, specified as (upperLeftCoord, lowerRightCoord)
nnImgSz = (80, 80)


scrnToNnInp scr = 
  -- Takes a raw uncroped screen string from ale and makes the full 
  -- transformation into an input ready for the neural network
  -- Param:
    -- scr tuple, (screen string, resolution as int tuple)
  -- Result:
    -- scr tuple, (screen as list of doubles, resolution as int tuple)

  -- Make hex list from string
  scr `seq` let hxLs = chunksOf 2 (C.unpack scr) in
  
  -- Crop 33 rows off of Top and 17 rows off of bottom
  hxLs `seq` let crpHxLs = drop (33 * 160) (take (193 * 160) hxLs) in
  
  -- Downsize to nnImgSz, XXX: Better Resize
  crpHxLs `seq` let colDropNnHxLs = map head (chunksOf 2 crpHxLs) in
  colDropNnHxLs `seq` let rowDropNnHxLs = foldl (++) [] (map head (chunksOf 2 (chunksOf 80 colDropNnHxLs))) in

  -- Convert to Grayscale
  rowDropNnHxLs `seq` let grayImg = VU.fromList [magicArray R.! (R.Z R.:. ((hTD (hex!!1))::Int) R.:. ((hTD  (hex!!0))::Int)) | hex <- rowDropNnHxLs] in

  rowDropNnHxLs `seq` grayImg


magicArray = 
  R.fromListUnboxed (R.Z R.:. (8::Int) R.:. (16::Int)) magicNumbers
  where magicNumbers = [0.0, 62.56, 51.92, 44.76, 28.56, 31.64, 23.52, 13.440000000000001, 9.520000000000001, 25.72, 37.68, 45.67999999999999, 42.599999999999994, 43.96, 43.32, 42.68, 63.36, 93.12, 77.4, 70.52000000000001, 57.72, 60.239999999999995, 52.959999999999994, 43.44, 39.519999999999996, 55.72, 68.24, 76.24, 74.27999999999999, 78.19999999999999, 74.71999999999998, 73.80000000000001, 106.91999999999999, 123.96, 100.03999999999999, 96.27999999999999, 83.76, 85.71999999999998, 79.28, 70.6, 69.52, 83.16000000000001, 95.68, 106.8, 105.96, 108.75999999999999, 105.0, 104.92, 142.56, 150.83999999999997, 125.52, 119.2, 108.96, 110.36, 104.75999999999999, 97.75999999999999, 95.55999999999999, 109.47999999999999, 122.56, 136.51999999999998, 136.51999999999998, 136.2, 132.44, 132.07999999999998, 174.23999999999998, 173.75999999999996, 144.2, 141.0, 131.04, 132.16, 127.4, 120.12, 118.76, 132.68, 146.04, 160.0, 160.28, 162.79999999999998, 159.04, 155.55999999999997, 198.0, 196.96, 162.88, 159.96, 153.12, 150.84, 146.92, 143.32, 141.12, 152.2, 168.68, 185.76, 186.88000000000002, 186.28, 182.52, 179.04, 217.79999999999998, 219.88, 181.28, 177.8, 174.35999999999999, 171.51999999999998, 168.44, 165.4, 163.20000000000002, 174.56, 191.32000000000002, 205.56, 207.79999999999998, 209.76000000000002, 202.88, 202.24, 233.64000000000001, 239.11999999999998, 199.96, 196.76, 193.32, 190.2, 187.12, 184.92000000000002, 182.71999999999997, 194.07999999999998, 211.12, 228.2, 230.44, 232.39999999999998, 225.51999999999998, 221.76] :: [Double]

-- XXX make this function more idomatic
hTD h = 
  -- Maps a hex digit to a decimal integer
  case h of '0' -> 0
            '1' -> 1
            '2' -> 2
            '3' -> 3
            '4' -> 4
            '5' -> 5
            '6' -> 6
            '7' -> 7
            '8' -> 8
            '9' -> 9
            'A' -> 10
            'B' -> 11
            'C' -> 12
            'D' -> 13
            'E' -> 14
            'F' -> 15
            _ -> error ("Value error on input: " ++ [h])
-- ##