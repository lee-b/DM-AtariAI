DM-AtariAI
==========
A Haskell implementation of the Deep Mind Atari AI

##Sub Components

- [X] Strict named pipe interface
- [X] Frame preprocessing
- [X] Neural Network Initilaization
- [X] Neural Network Output function
- [ ] Neural Network Training function
- [X] CPU 4D tensor convolution with stride paramter
- [ ] GPU 4D tensor convolution with stride parameter


##Running

```ghc -O2 -threaded Main.hs```

```./Main +RTS -N```

##Dependencies

ALE (Arcade Learning Environement) and its dependencies
