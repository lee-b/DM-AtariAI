-- Author: Hamzeh Alsalhi <93hamsal@gmail.com>

module Main where
  import System.Process
  import System.Posix.Files
  import System.Posix.IO
  import System.IO
  import qualified Data.Vector as V
  import Control.Concurrent
  import System.Directory

  mem_sz = 50

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
      case (l >= mem_sz) of True ->  foreverPrint hndl_in fout 
                                      (str : init mem) l
                            False -> foreverPrint hndl_in fout 
                                      (str : mem) (l + 1)