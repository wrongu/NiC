--[[
based on Google's convnet_atari_3
]]

require 'convnet'

return function(args)
	-- note that these shapes depend on iniput dimensions. Should be arranged so that
	-- there is nothing truncated by integer division
	-- output width = (input width - filter width) / (filter stride) + 1
	-- currently set for 28x28 maps such that layers become
	-- 32x9x9    (to be pedantic, (28 - 8) / 4 + 1 = 6)
	-- 64x2x2
	-- 64x1x1
    args.n_units        = {32, 64, 64}
    args.filter_size    = {8, 3, 2}
    args.filter_stride  = {4, 3, 1}
    args.n_hid          = {512}
    args.nl             = nn.Rectifier

    return create_network(args)
end

