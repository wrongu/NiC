Ants AI Challenge with DQN
--------------------------

The goal of this project is to test the DQN framework ("Human-level control through deep reinforcement learning", Nature 518, 529–533 (26 February 2015) doi:10.1038/nature14236.) in a multi-agent environment. The environment is the [Ants AI challenge](http://ants.aichallenge.org/), hosted by Google in 2012.

The LICENSE here applies to the following files (exact or slightly modified code from Google).

	convnet_atari3.lua
	net_downsample_2x_full_y.lua
	nnutils.lua
	Rectifier.lua
	train_agent.lua
	convnet.lua
	initenv.lua
	NeuralQLearner.lua
	Scale.lua
	TransitionTable.lua

To run, requires a `torch` directory inside `dqnbot/`. On Linux (after [installing torch](http://torch.ch/docs/getting-started.html#_):

	ln -s ~/torch ./dqnbot/torch

