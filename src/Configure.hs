module Configure where

-- Choose ROM and actions available to the AI in the game being played
rom = ["space_invaders.bin", "breakout.bin"] !! 1
availActns = [['0', '1', '3', '4'], ['0', '1', '3', '4']] !! 1
numAvailActns = length availActns

frameSkip = 4
displayScreen = True

numGames = 1000

memSize = 300
 