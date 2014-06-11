--module Main where
 
--import qualified Data.Map as M
 
--errorsPerLine = M.fromList
--    [ ("Chris", 472), ("Don", 100), ("Simon", -5) ]
 
--main = do putStrLn "Who are you?"
--          name <- getLine
--          case M.lookup name errorsPerLine of
--              Nothing -> putStrLn "I don't know you"
--              Just n  -> do putStr "Errors per line: "
--                            print n


-- #############################################################

---- file: ch07/toupper-imp.hs
--import System.IO
--import Data.Char(toUpper)

--main :: IO ()
--main = do 
--       inh <- openFile "ale_fifo_out" ReadMode
--       outh <- openFile "ale_fifo_in" WriteMode
--       mainloop inh outh
--       hClose inh
--       hClose outh

--mainloop :: Handle -> Handle -> IO ()
--mainloop inh outh = 
--    do ineof <- hIsEOF inh
--       if ineof
--           then return ()
--           else do inpStr <- hGetLine inh
--                   hPutStrLn outh (map toUpper inpStr)
--                   mainloop inh outh

module Main where

import System.Process
import System.IO

import Control.Parallel
import Control.Concurrent
import System.Environment
import System.FilePath

import Control.Monad
import System.IO
import System.Posix.Files
import System.Posix.IO

main = do (readProcess "rm" ["ale_fifo_out"] "")
          (readProcess "rm" ["ale_fifo_in"] "")
          createNamedPipe "ale_fifo_out" $ unionFileModes ownerReadMode ownerWriteMode
          createNamedPipe "ale_fifo_in"  $ unionFileModes ownerReadMode ownerWriteMode
         
          let toA_IO = openFd "ale_fifo_in"   WriteOnly Nothing defaultFileFlags
          let fromA_IO = openFd "ale_fifo_out" ReadOnly  Nothing defaultFileFlags
          putStrLn "Hangtest 1"      
          
          -- Command 1) ../ale/ale_0.4.4/ale_0_4/ale -max_num_episodes 5 -game_controller fifo_named -disable_colour_averaging true -run_length_encoding false -frame_skip 4 -display_screen true ../ale/ale_0.4.4/ale_0_4/roms/space_invaders.bin

          fromA <- fromA_IO
          putStrLn "Hangtest 3"

          toA <- toA_IO   
          putStrLn "Hangtest 2"

          (str, _len) <- fdRead fromA 64
          putStrLn str
          putStrLn "Hangtest 4"
          
          fdWrite toA "1,0,0,1\n"
          putStrLn "Hangtest 5"

          foreverPrint fromA

foreverPrint f = do (str, _len) <- fdRead f 64
                    putStrLn str
                    foreverPrint f

-- June 11 - FIFO read from ALE works!
     -- run this
     -- start ale