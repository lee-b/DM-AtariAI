DM-AtariAI
==========
A Haskell implementation of the Deep Mind Atari AI. Work has been done here https://github.com/kristjankorjus/Replicating-DeepMind to build the bot in python and I am referring to it for ideas of the impementation.

## Base Features

- [X] Strict named pipe interface
- [X] Frame preprocessing
- [X] Neural Network Initilaization
- [X] Neural Network Output function
- [ ] Neural Network Training function
- [X] CPU 4D tensor convolution with stride paramter
- [x] GPU 4D tensor convolution with stride parameter


##Running

```ghc -O2 -threaded Main.hs```

```./Main +RTS -N```

##Dependencies

ALE (Arcade Learning Environement) and its dependencies


## Release - 0.1
!()[images/cpu-sat-v0.1]

## Release - 0.2
!()[images/cpu-sat-v0.1]
