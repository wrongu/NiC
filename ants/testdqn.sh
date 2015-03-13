#!/bin/sh

TURNS=1000
START_GENERATION=$1
END_GENERATION=$2

for GEN in `seq $START_GENERATION $END_GENERATION`
do
	echo "python tools/playgame.py \"dqnbot/torch/bin/luajit DQNBot.lua -resume -save_name hunter."$TURNS"."$GEN" -nturns "$TURNS"\" \"python tools/sample_bots/python/HunterBot.py\" --nolaunch --map_file tools/maps/example/tutorial1.map --log_dir game_logs --turns $TURNS --scenario --food none --player_seed 7 --verbose -e"
	python tools/playgame.py "dqnbot/torch/bin/luajit DQNBot.lua -resume -save_name hunter."$TURNS"."$GEN" -nturns "$TURNS"" "python tools/sample_bots/python/HunterBot.py" --nolaunch --map_file tools/maps/example/tutorial1.map --log_dir game_logs --turns $TURNS --scenario --food none --player_seed 7 --verbose -e
done