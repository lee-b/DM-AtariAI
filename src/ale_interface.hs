-- Author: Hamzeh Alsalhi <93hamsal@gmail.com>

module Main where
import System.Process
import System.Posix.Files
import System.Posix.IO
import System.IO
import Control.Concurrent
import System.Directory
import qualified Data.Vector as V
import Data.List.Split
import qualified Data.Vector.Unboxed
import qualified Data.Array.Repa as R

memSz = 50
aleScrnSz = (160, 210)

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

    let memory = []
    let toA_IO = openFd "ale_fifo_in"   WriteOnly Nothing 
                  defaultFileFlags
    let fromA_IO = openFd "ale_fifo_out" ReadOnly  Nothing 
                  defaultFileFlags  

    fromA <- fromA_IO
    toA <- toA_IO

    (str, _len) <- fdRead fromA (67206)
    putStrLn str
    fdWrite toA "1,0,0,1\n"

    (str, _len) <- fdRead fromA (67206)
    putStrLn str
    -- first action must be 1 to satrt game
    fdWrite toA "1,18\n"

    hndl_in <- fdToHandle fromA
    foreverPrint hndl_in toA memory 0

foreverPrint hndl_in fout mem l = 
 do str <- hGetLine hndl_in
    --let mem_n = mem V.++ (V.singleton str)
    fdWrite fout "0,18\n"
    putStrLn ("Screen string " ++ (take 8 (reverse (take 10 (reverse str))))
      ++  "... has length " ++ (show (length str)))
    --putStrLn (show (V.length mem_n))
    putStrLn (show (length mem))
    case (l >= memSz) of  True ->  foreverPrint hndl_in fout
                                    (str : init mem) l
                          False -> foreverPrint hndl_in fout
                                    (str : mem) (l + 1)

-- Screen space interested in, specified as (upperLeftCoord, lowerRightCoord)
roi = ((0, 33),(160,193))
nnImgSz = (80, 80)

scrnToNnInp scr = 
  -- Takes a raw uncroped screen string from ale and makes the full 
  -- transformation into an input ready for the neural network

  -- Param:
    -- scr tuple,  (screen string, resolution int tuple)
 
  -- XXX: Assert length of screen string is even

  -- Make hex list from string
  let hxLs = (chunksOf 2 (fst scr), (snd scr)) in
  -- Crop 33 rows off of Top and 17 rows off of bottom
  let crpHxLs = (drop (33 * fst $ snd hxLs)  (take (193 * fst $ snd hxLs) (fst hxLs)) , (160, 160)) in
  -- Downsize to nnImgSz
  let nnHxLs = (map head (chunksOf 2 (fst crpHxLs)), (80,80))in
  -- XXX: Better resize function if necessary, this hack only drops evey other
  -- and will not give a good quality result http://stackoverflow.com/questions
  -- /6133957/image-downsampling-algorithms

  -- XXX: Convert to Grayscale

  []

sclImg img (xdim, ydim) =
  -- scales img to become size xdim ydim
  []

clrToBw = 
  R.fromListUnboxed (R.Z R.:. (5::Int) R.:. (2::Int)) magicNumbers
  where magicNumbers = [0.0, 62.56, 51.92, 44.76, 28.56, 31.64, 23.52, 13.440000000000001, 9.520000000000001, 25.72, 37.68, 45.67999999999999, 42.599999999999994, 43.96, 43.32, 42.68, 63.36, 93.12, 77.4, 70.52000000000001, 57.72, 60.239999999999995, 52.959999999999994, 43.44, 39.519999999999996, 55.72, 68.24, 76.24, 74.27999999999999, 78.19999999999999, 74.71999999999998, 73.80000000000001, 106.91999999999999, 123.96, 100.03999999999999, 96.27999999999999, 83.76, 85.71999999999998, 79.28, 70.6, 69.52, 83.16000000000001, 95.68, 106.8, 105.96, 108.75999999999999, 105.0, 104.92, 142.56, 150.83999999999997, 125.52, 119.2, 108.96, 110.36, 104.75999999999999, 97.75999999999999, 95.55999999999999, 109.47999999999999, 122.56, 136.51999999999998, 136.51999999999998, 136.2, 132.44, 132.07999999999998, 174.23999999999998, 173.75999999999996, 144.2, 141.0, 131.04, 132.16, 127.4, 120.12, 118.76, 132.68, 146.04, 160.0, 160.28, 162.79999999999998, 159.04, 155.55999999999997, 198.0, 196.96, 162.88, 159.96, 153.12, 150.84, 146.92, 143.32, 141.12, 152.2, 168.68, 185.76, 186.88000000000002, 186.28, 182.52, 179.04, 217.79999999999998, 219.88, 181.28, 177.8, 174.35999999999999, 171.51999999999998, 168.44, 165.4, 163.20000000000002, 174.56, 191.32000000000002, 205.56, 207.79999999999998, 209.76000000000002, 202.88, 202.24, 233.64000000000001, 239.11999999999998, 199.96, 196.76, 193.32, 190.2, 187.12, 184.92000000000002, 182.71999999999997, 194.07999999999998, 211.12, 228.2, 230.44, 232.39999999999998, 225.51999999999998, 221.76] :: [Double]