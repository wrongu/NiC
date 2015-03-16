#!/bin/sh
NET=$1
EP=$2
TURNS=60

# Create a new random map for this trial
tools/mapgen/mapgen.py --no_players=2 --max_hills=1 --min_dimensions=30 --max_dimensions=40 > tmpmap
python tools/playgame.py "dqnbot/torch/bin/luajit DQNBot.lua -resume -test -testing_ep "$EP" -save_name "$NET" -nturns "$TURNS"" "python tools/sample_bots/python/HunterBot.py" --map_file tmpmap --log_dir game_logs --turns $TURNS -e
