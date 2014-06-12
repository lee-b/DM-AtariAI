module Main where
import System.Process
import System.Posix.Files
import System.Posix.IO

main = do (readProcess "rm" ["ale_fifo_out"] "")
          (readProcess "rm" ["ale_fifo_in"] "")
          createNamedPipe "ale_fifo_out" $ unionFileModes ownerReadMode ownerWriteMode
          createNamedPipe "ale_fifo_in"  $ unionFileModes ownerReadMode ownerWriteMode
         
          let toA_IO = openFd "ale_fifo_in"   WriteOnly Nothing defaultFileFlags
          let fromA_IO = openFd "ale_fifo_out" ReadOnly  Nothing defaultFileFlags
          putStrLn "Hangtest 1"      
          
          -- Command 1)  ./ale_interface > output.txt & ../ale/ale_0.4.4/ale_0_4/ale -max_num_episodes 5 -game_controller fifo_named -disable_colour_averaging true -run_length_encoding false -frame_skip 0 -display_screen true ../ale/ale_0.4.4/ale_0_4/roms/space_invaders.bin

          fromA <- fromA_IO
          putStrLn "Hangtest 3"

          toA <- toA_IO   
          putStrLn "Hangtest 2"

          (str, _len) <- fdRead fromA (160 * 210 * 2 + 200)
          putStrLn str
          putStrLn "Hangtest 4"
          
          fdWrite toA "1,0,0,1\n"
          putStrLn "Hangtest 5"

          (str, _len) <- fdRead fromA (160 * 210 * 2 + 200)
          putStrLn str
          fdWrite toA "1,18\n" -- first action must be 1,1 to satrt game

          foreverPrint fromA toA

foreverPrint fin fout = do (str, _len) <- fdRead fin (160 * 210 * 2 + 200)
                           fdWrite fout "3,18\n"
                           --x <- getChar
                           putStrLn str
                           putStrLn "####"
                           foreverPrint fin fout

-- June 11 - FIFO read from ALE works!
     -- run this
     -- start ale