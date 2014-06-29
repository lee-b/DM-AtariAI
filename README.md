DM-AtariAI
==========
A Haskell implementation of the Deep Mind Atari AI

##Sub Components

Arcade Learning Environment (ALE) interface written in hasekll A strict implementation of a FIFO interface using the ByteString package

4D tensors Neural network convolution Implements convolutional layer given a vanilla 2D signal convolution function

##Running

'''ghc -O2 -prof -auto-all ale_interface.hs'''

'''./ale_interface'''

'''../ale/ale_0.4.4/ale_0_4/ale -max_num_episodes 5 -game_controller fifo_named -disable_colour_averaging true -run_length_encoding false -frame_skip 1 -display_screen true ../ale/ale_0.4.4/ale_0_4/roms/space_invaders.bin'''

##Dependencies

ALE

-XXX: graphics libraries

Haskell Platform

Repa
