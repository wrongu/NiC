--[[
preprocess_rescale: rescales the given row x col x 4 raw state into canonical
width x height x 4 using bilinear image interpolation

99% based on net_downsample_2x_full_y.lua and Scale.lua (both copyright by Google - see sources)
--]]

require "nn"
require "image"

local scale = torch.class('nn.Scale', 'nn.Module')

function scale:__init(height, width)
    self.height = height
    self.width = width
end

function scale:forward(x)
    local x = image.scale(x, self.width, self.height, 'bilinear')
    return x
end

function scale:updateOutput(input)
    return self:forward(input)
end

function scale:float()
end


local function create_network(args)
    -- Y (luminance)
    return nn.Scale(args.layer_1_width, args.layer_1_width, true)
end

return create_network
