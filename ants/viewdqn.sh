#!/bin/sh
NET=$1
EP=$2
TURNS=60

python tools/playgame.py "dqnbot/torch/bin/luajit DQNBot.lua -resume -test -testing_ep "$EP" -save_name "$NET" -nturns "$TURNS"" "python tools/sample_bots/python/HunterBot.py" --map_file tools/maps/example/tutorial1.map --turntime 5000 --log_dir game_logs --turns $TURNS --scenario --food none --player_seed 7 --verbose -e
