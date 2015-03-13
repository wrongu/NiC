--[[
preprocess_toroidal_tile.lua: resizes 3d tensor in 2nd and 3rd dimensions, cropping if too large and
filling with torus topology if too small

90% based on net_downsample_2x_full_y.lua and Scale.lua (both copyright by Google - see sources)
--]]

require "nn"
require "torch"

local tile = torch.class('nn.Tile', 'nn.Module')

function tile:__init(height, width)
    self.height = height
    self.width = width
end

function tile:prepareTiles(input)
	-- allocate tensor on first pass only
	local sz = input:size()
	if not self.tiles then
		assert(input:dim() == 3)
		self.gradInput = torch.Tensor():resizeAs(input):zero()

	    local n_row_tiles = math.ceil(sz[2] / self.height) + 2
	    local n_col_tiles = math.ceil(sz[3] / self.width) + 2
	    self.tiles = torch.Tensor(sz[1], sz[2]*n_row_tiles, sz[3]*n_col_tiles):zero()

	    -- slice indices into tile tensor such that center of input is center of output
	    local cr = math.floor(n_row_tiles / 2) * sz[2] + math.floor(sz[2] / 2)
	    local cc = math.floor(n_col_tiles / 2) * sz[3] + math.floor(sz[3] / 2)
	    -- compute bounds: half width in each direction from center
	    self.row_slice = {math.ceil(cr - self.height / 2), math.ceil(cr + self.height / 2 - 1)}
	    self.col_slice = {math.ceil(cc - self.width / 2), math.ceil(cc + self.width / 2 - 1)}
	end

	-- copy input into tiles
	for r = 1, self.tiles:size()[2], sz[2] do
		for c = 1, self.tiles:size()[3], sz[3] do
			self.tiles:sub(1, sz[1], r, r+sz[2]-1, c, c+sz[3]-1):copy(input)
		end
	end
end

function tile:updateOutput(input)
	self:prepareTiles(input)
	self.output = self.tiles:sub(1, input:size()[1], self.row_slice[1], self.row_slice[2], self.col_slice[1], self.col_slice[2])
	return self.output
end

function tile:updateGradInput(input, gradOutput)
	return self.gradInput
end

function tile:float()
	return self.output
end

local function create_network(args)
    return nn.Tile(args.layer_1_width, args.layer_1_width)
end

return create_network
