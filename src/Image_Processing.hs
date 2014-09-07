module Image_Processing where

import qualified Data.Array.Repa as R
import qualified Data.ByteString.Char8 as C
import qualified Data.Vector.Unboxed as VUN
import Data.List.Split

scrnToNnInp scr = 
  -- Takes a raw uncroped screen string from ale and makes the full 
  -- transformation into an input ready for the neural network
  -- Param:
    -- scr screen string
  -- Result:
    -- scr screen as list of doubles

  -- Make hex list from string
  let hxLs = chunksOf 2 (C.unpack scr)
  -- Crop 33 rows off of Top and 17 rows off of bottom
      crpHxLs = drop (33 * 160) (take (193 * 160) hxLs)
  -- Downsize to nnImgSz: 80,by 80, XXX: Better Resize
      colDropNnHxLs = map head (chunksOf 2 crpHxLs)
      rowDropNnHxLs = foldl (++) [] 
                        (map head (chunksOf 2 (chunksOf 80 colDropNnHxLs)))
  -- XXX Pad with blank pixels from 80,80 to 84,84
      rowDropNnHxLsPded = rowDropNnHxLs ++ ["00" | x <- [1..(84*84 - 80*80)]]
  -- Convert to Grayscale
      grayImg = VUN.fromList [magicArray R.! (R.Z R.:. ((hTD (hex!!1))::Int) R.:. ((hTD  (hex!!0))::Int)) | hex <- rowDropNnHxLsPded]
  in grayImg

magicArray = (R.fromListUnboxed (R.Z R.:. (8::Int) R.:. (16::Int)) (magicNumbers  :: [Float]))
  where magicNumbers = [0.0, 62.56, 51.92, 44.76, 28.56, 31.64, 23.52, 13.440000000000001, 9.520000000000001, 25.72, 37.68, 45.67999999999999, 42.599999999999994, 43.96, 43.32, 42.68, 63.36, 93.12, 77.4, 70.52000000000001, 57.72, 60.239999999999995, 52.959999999999994, 43.44, 39.519999999999996, 55.72, 68.24, 76.24, 74.27999999999999, 78.19999999999999, 74.71999999999998, 73.80000000000001, 106.91999999999999, 123.96, 100.03999999999999, 96.27999999999999, 83.76, 85.71999999999998, 79.28, 70.6, 69.52, 83.16000000000001, 95.68, 106.8, 105.96, 108.75999999999999, 105.0, 104.92, 142.56, 150.83999999999997, 125.52, 119.2, 108.96, 110.36, 104.75999999999999, 97.75999999999999, 95.55999999999999, 109.47999999999999, 122.56, 136.51999999999998, 136.51999999999998, 136.2, 132.44, 132.07999999999998, 174.23999999999998, 173.75999999999996, 144.2, 141.0, 131.04, 132.16, 127.4, 120.12, 118.76, 132.68, 146.04, 160.0, 160.28, 162.79999999999998, 159.04, 155.55999999999997, 198.0, 196.96, 162.88, 159.96, 153.12, 150.84, 146.92, 143.32, 141.12, 152.2, 168.68, 185.76, 186.88000000000002, 186.28, 182.52, 179.04, 217.79999999999998, 219.88, 181.28, 177.8, 174.35999999999999, 171.51999999999998, 168.44, 165.4, 163.20000000000002, 174.56, 191.32000000000002, 205.56, 207.79999999999998, 209.76000000000002, 202.88, 202.24, 233.64000000000001, 239.11999999999998, 199.96, 196.76, 193.32, 190.2, 187.12, 184.92000000000002, 182.71999999999997, 194.07999999999998, 211.12, 228.2, 230.44, 232.39999999999998, 225.51999999999998, 221.76]

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