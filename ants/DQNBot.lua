--[[
DQNBot

written by Richard Lange
March 2015

Google AI Challenge 2012 (Ants) bot using the neural-Q-learning algorithm
from Minh et al 2015
--]]

local ants  = require "Ants"

package.path = package.path .. ';dqnbot/?.lua'

require 'torch'
require 'initenv'
require 'nn'
require 'nngraph'
require 'nnutils'
require 'image'
require 'NeuralQLearner'
require 'TransitionTable'
require 'Rectifier'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-network', 'convnet_ants', 'network module or .t7 file')
cmd:option('-seed', os.time(), 'random seed')
cmd:option('-test', false, 'if testing, no save and no learning')
cmd:option('-testing_ep', 0.95, 'percent greediness in testing mode')
cmd:option('-save_dir', 'dqnbot/saved_networks', 'directory to save snapshots')
cmd:option('-save_name', '', 'filename for saving networks, ending in .<generation i>')
cmd:option('-resume', false, 'whether to load i-1th network')
cmd:option('-save_freq', 200, 'frequency to overwrite save file (dont know when game is over)')
cmd:option('-nturns', 1000, 'max # turns in this game (so we know when game is over)')

cmd:text()

local opt = cmd:parse(arg)

-- Scoring for reinforcement learning
local VALUE = {
	ANT = 1,
	HILL = 100
}

--[[
args to nql:init are as follows:
 - state_dim         dimensionality of state space
 - actions           list of actions
 - verbose           flag: verbosity
 - best              flag: use best learned network
 - ep                epsilon annealing starting value (1=explore, 0=exploit)
 - ep_end            final epsilon value (i.e. percent random play by expert network)
 - ep_endt           number of steps until reached ep_end
 - lr                initial learning rate
 - lr_end            final learning rate
 - lr_endt           number of steps to reach lr_end
 - wc                L2 weight cost (regularization)
 - minibatch_size    learn (S,A) pairs in batches
 - valid_size        size of set to compute statistics on
 - discount          'gamma' of TD learning
 - update_freq       number of steps between minibatch updates
 - n_replay          number of learning steps per update
 - learn_start       frame # where learning starts (must be larger than batch size)
 - replay_memory     max size of history table
 - hist_len          number of frames processed for current turn (set to 4 in paper)
 - rescale_r         whether reward should be clamped/rescaled
 - max_reward        max +reward
 - min_reward        min -reward
 - clip_delta        clip targets to +/- delta (set to nil to leave unbounded)
 - target_q          number of steps between updating target network Qhat
 - gpu               flag: use CUDA
 - layer_1_width     first convolutional layer is (hist_len*ncols) x layer_1_width x layer_1_width (default is 84)
 - ncols             number of colors in input (used in constructing default W x W x colors x hist_len input layer)
 - input_dims        number of inputs into network (Default W x W x colors x hist_len)
 - preproc           name of lua module that returns a function. This function must return a nn.Module instance
 - histType          'linear' 'exp2' or 'exp1.25': how to sample the hist_len processed frames from all history
 - histSpacing       if histType is linear, sample-able history is every `histSpacing`th frame
 - nonTermProb       Discard non-terminal states from minibatch sample with probability (1-nonTermProb)
 - bufferSize        size of history buffer from which replays are sampled (must be bigger than minibatch_size)
 - transition_params unused
 - network           name of network file (either lua module that returns a createNetwork() function or filename of torch saved network)
--]]

-- dqnbot is a subclass of NeuralQLearner
local dqnbot = torch.class("DQNBot", "dqn.NeuralQLearner")

function dqnbot:__init(args)
	dqn.NeuralQLearner:__init(args)
end

local bot = {}

function bot:onReady()
	-- initialization
	args = {}
	args.actions        = {"N", "S", "E", "W", "C"} -- an ant may move NSEW or stay still (C for center)
	args.verbose        = 2
	args.best           = true
	args.ep             = 1
	args.ep_end         = 0.05
	args.ep_endt        = math.floor(opt.nturns*0.9)
	args.lr             = 0.00025
	args.lr_endt        = args.ep_endt -- learning stops at the same time we switch into "expert" mode
	args.wc             = 0 -- no normalization
	args.minibatch_size = 5
	args.valid_size     = 50
	args.discount       = 0.99 -- looking far into the future
	args.update_freq    = 4
	args.n_replay       = 6
	args.learn_start    = 100 -- must be larger than max(minibatch size, validation size, bufferSize)
	args.replay_memory  = args.ep_endt -- no need to remember farther back than # steps used for learning
	args.hist_len       = 3 -- make decisions based on last 3 frames
	args.layer_1_width  = 40
	args.rescale_r      = true
	args.max_reward     = 1
	args.min_reward     = -1
	args.clip_delta     = 1
	args.target_q       = 100 -- update target Q every 100 steps (should be larger??)
	args.gpu            = -1
	args.ncols          = 4 -- ants bot has a color channel for {land, ants, food, hills}
	args.preproc        = "dqnbot.preprocess_toroidal_tile"
	args.bufferSize     = 64
	args.network        = opt.network -- defaults to "convent_atari3", otherwise loads from -network command line option
	args.state_dim      = args.ncols * args.layer_1_width * args.layer_1_width
	args.seed           = opt.seed

	-- load last network if specified
	if opt.resume then
		local base_name, generation = string.match(opt.save_name, "(.+)%.(%d)$")
		if generation then
			generation = tonumber(generation)
			if generation > 1 then
				args.network = opt.save_dir .. "/" .. base_name .. "." .. (generation-1) .. ".t7"
				io.stderr:write(args.network .. "\n"); io.stderr:flush()
			end
		end
	end

	-- create agent
	torchSetup(args)
	self.dqn_agent = DQNBot(args)
	self.reward_history = {}
	self.testing = opt.test
	self.testing_ep = opt.testing_ep

	-- store some key info in self
	self.image_width = layer_1_width
	self.valid_actions = args.actions
	self.score = VALUE.ANT -- start with a score of 1
	self.ncols = args.ncols

	-- tell engine we're ready
	ants:finishTurn()
end

function bot:map_to_tensor()
	local rows = ants.config.rows
	local cols = ants.config.cols

	if not self.tensor_map then
		self.tensor_map = torch.Tensor(self.ncols, rows, cols)
		self.map2x2 = torch.Tensor(self.ncols, 2*rows, 2*cols)
	end
	-- clear values from previous update
	self.tensor_map:zero()

	-- layers are as follows:
	-- [1, :, :] = terrain { -1 water, 1 land}
	-- [2, :, :] = ants {-1 enemy, 0 empty, 1 ally}
	-- [3, :, :] = food {0 empty, 1 food}
	-- [4, :, :] = hills {-1 enemy, 0 empty, 1 ally}

	for r = 0, rows-1 do
		for c = 0, cols-1 do
			-- mark LAND and WATER
			if ants.map[r][c] == ants.landTypes.WATER then
				self.tensor_map[1][r+1][c+1] = -1
			else
				self.tensor_map[1][r+1][c+1] = 1
			end
		end
	end

	-- each other map (ants, food, hills) is stored in an array
	-- todo (?) abstract this into table {ant=2, food=3, hill=4}
	for _,ant in ipairs(ants.ants) do
		local team = ant.owner == 0 and 1 or -1 -- ternary conditional
		self.tensor_map[2][ant.row+1][ant.col+1] = team
	end

	for _,food in ipairs(ants.food) do
		self.tensor_map[3][food.row+1][food.col+1] = 1
	end

	for _,hill in ipairs(ants.hills) do
		local team = hill.owner == 0 and 1 or -1 -- ternary conditional
		self.tensor_map[4][hill.row+1][hill.col+1] = team
	end

	-- tile the map into 2x2 (makes toroidal topology easier to index later)
	self.map2x2:sub(1, self.ncols, 1,      rows,   1,      cols):copy(self.tensor_map)
	self.map2x2:sub(1, self.ncols, 1,      rows,   cols+1, 2*cols):copy(self.tensor_map)
	self.map2x2:sub(1, self.ncols, rows+1, 2*rows, 1,      cols):copy(self.tensor_map)
	self.map2x2:sub(1, self.ncols, rows+1, 2*rows, cols+1, 2*cols):copy(self.tensor_map)

	return self.tensor_map
end

function bot:egocentric_map(row, col)
	--[[
		translate the map so that the map coordinate (row, col) is in the center.
			  c
		aaaaaa|bb      aaaa|bbaa  
		aaaaaa|bb      aaaa|bbaa
		aaaaaa|bb  ->  ----+----
		------+-- r    cccc|ddcc
		cccccc|dd      aaaa|bbaa

		This is done by tiling the original map into a 2x2 grid then copying the
		appropriate sub-map out of the larger map
	--]]


	local rows = ants.config.rows
	local cols = ants.config.cols
	-- center coordinate
	local cr = math.floor(rows / 2)
	local cc = math.floor(cols / 2)
	-- delta
	local dr = cr - row
	local dc = cc - col
	-- indices in map2x2 from which (row x col) egocentric map is copied
	local row_slice = {1-dr, rows-dr}
	local col_slice = {1-dc, cols-dc}

	if row_slice[1] < 1 then
		row_slice = {row_slice[1] + rows, row_slice[2] + rows}
	end
	if col_slice[1] < 1 then
		col_slice = {col_slice[1] + cols, col_slice[2] + cols}
	end

	return self.map2x2:sub(1, self.ncols, row_slice[1], row_slice[2], col_slice[1], col_slice[2])
end

function bot:compute_score()
	return #ants:myAnts() * VALUE.ANT + #ants:myHills() * VALUE.HILL
end

function bot:save()
	-- save network parameters and history (copied from train_agent.lua, part of the original DQN code)
	local agent = self.dqn_agent
	local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
		agent.valid_s2, agent.valid_term
	agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
		agent.valid_term = nil, nil, nil, nil, nil, nil, nil
	local w, dw, g, g2, delta, delta2, deltas, tmp = agent.w, agent.dw,
		agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp
	agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
		agent.deltas, agent.tmp = nil, nil, nil, nil, nil, nil, nil, nil

	local filename = opt.save_dir .. "/" .. opt.save_name
	torch.save(filename..'.t7', {agent = agent,
						model = agent.network,
						best_model = agent.best_network,
						reward_history = self.reward_history})
	local nets = {network=w:clone():float()}
	torch.save(filename..'.params.t7', nets, 'ascii')
	agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
		agent.valid_term = s, a, r, s2, term
	agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
		agent.deltas, agent.tmp = w, dw, g, g2, delta, delta2, deltas, tmp
	io.flush()
	collectgarbage()
end

function bot:onTurn()
	-- state updates already happened in Ants.lua
	-- update tensor map (i.e. translate from ants grid to a torch tensor)
	self:map_to_tensor()

	-- For each ant, get egocentric map of the world then perceive() and act
	local myAnts = ants:myAnts()

	local current_score = self:compute_score()
	local delta_score = current_score - self.score
	self.score = current_score
	table.insert(self.reward_history, delta_score)

	-- note: can't see enemy hills, so counting them will be 0 at the start
	local game_over = #ants:myHills() == 0 or ants.currentTurn > opt.nturns

	-- DEBUGGING
	-- if ants.currentTurn == 15 then
	--  for i,ant in ipairs(myAnts) do
	--      m = self:egocentric_map(ant.row, ant.col)
	--      image.save('test/ant.' .. i .. '.land.png',  m[1]:add(1):mul(0.5))
	--      image.save('test/ant.' .. i .. '.ants.png',  m[2]:add(1):mul(0.5))
	--      image.save('test/ant.' .. i .. '.food.png',  m[3]:add(1):mul(0.5))
	--      image.save('test/ant.' .. i .. '.hills.png', m[4]:add(1):mul(0.5))
	--  end
	-- end

	for _,ant in ipairs(myAnts) do
		m = self:egocentric_map(ant.row, ant.col)
		local action_idx = self.dqn_agent:perceive(delta_score, m, game_over, self.testing, self.testing_ep)
		local action = self.valid_actions[action_idx]
		if action and action ~= "C" and ants:passable(ant.row, ant.col, action) then
			ants:issueOrder(ant.row, ant.col, action)
		end
	end

	if not self.testing and game_over or ants.currentTurn >= opt.save_freq and ants.currentTurn % opt.save_freq == 0 then
		self:save()
	end

	ants:finishTurn()
end

function contains(t, k)
	return t[k] ~= nil
end

function bot:onEnd()
end


ants:start(bot)
