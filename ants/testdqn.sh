#!/bin/sh

TURNS=1000
START_GENERATION=$1
END_GENERATION=$2
PREFIX=$3

for GEN in `seq $START_GENERATION $END_GENERATION`
do
	FILENAME="$PREFIX.$TURNS.$GEN"
	TRIES=0
	# Note that if it doesn't reach save_freq turns, no network is saved.
	# So, repeat until it works:
	while [ ! -e "dqnbot/saved_networks/$FILENAME.t7" ] && [ $TRIES -lt 30 ]
	do
		echo "attempt $TRIES"
		echo "python tools/playgame.py \"dqnbot/torch/bin/luajit DQNBot.lua -resume -save_freq 100 -save_name "$FILENAME" -nturns "$TURNS"\" \"python tools/sample_bots/python/HunterBot.py\" --nolaunch --turntime 5000 --map_file tools/maps/example/tutorial1.map --log_dir game_logs --turns $TURNS --scenario --food none --player_seed 7 --verbose -e"	
		python tools/playgame.py "dqnbot/torch/bin/luajit DQNBot.lua -resume -save_freq 100 -save_name "$FILENAME" -nturns "$TURNS"" "python tools/sample_bots/python/HunterBot.py" --nolaunch --turntime 5000 --map_file tools/maps/example/tutorial1.map --log_dir game_logs --turns $TURNS --scenario --food none --player_seed 7 --verbose -e
		let TRIES=TRIES+1
	done

	# if still unsuccessful, just quit
	if [ ! -e "dqnbot/saved_networks/$FILENAME.t7" ]
	then
		break;
	fi
done

echo "-DONE-"
