module ALE_Interface where

import System.Directory
import System.Process
import System.Posix.Files
import System.Posix.IO
import qualified Data.ByteString.Char8          as C
import qualified Configure                      as CF

initialize = 
 do fo_ex <- doesFileExist "ale_fifo_out"
    fi_ex <- doesFileExist "ale_fifo_in"

    case fo_ex of True -> (readProcess "rm" ["ale_fifo_out"] "")
                  False -> (readProcess "ls" [] "")
    case fi_ex of True -> (readProcess "rm" ["ale_fifo_in"] "")
                  False -> (readProcess "ls" [] "")

    let fMode = unionFileModes ownerReadMode ownerWriteMode
    createNamedPipe "ale_fifo_out" fMode
    createNamedPipe "ale_fifo_in" fMode

    -- Run ALE
    runCommand ("../ale/ale_0.4.4/ale_0_4/ale \
                \-max_num_episodes 0 \
                \-game_controller fifo_named \
                \-disable_colour_averaging true \
                \-run_length_encoding false \
                \-frame_skip " ++ show CF.frameSkip ++ " \
                \-display_screen " ++ show (if CF.displayScreen then "true"
                                            else "false") ++ " \
                \../ale/ale_0.4.4/ale_0_4/roms/" ++ show CF.rom)

    fromA <- openFd "ale_fifo_out" ReadOnly Nothing defaultFileFlags
    fromA <- fdToHandle fromA
    toA <- openFd "ale_fifo_in" WriteOnly Nothing defaultFileFlags

    -- Send handsakes string to ALE, Instruct we want screen and epsiode info
    str <- C.hGetLine fromA
    fdWrite toA "1,0,0,1\n"

    return (toA, fromA)
