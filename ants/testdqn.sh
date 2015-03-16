#!/bin/sh

TURNS=1000
START_GENERATION=$1
END_GENERATION=$2
PREFIX="hunter_rand"

for GEN in `seq $START_GENERATION $END_GENERATION`
do
	FILENAME="$PREFIX.$TURNS.$GEN"
	TRIES=0
	# Create a new random map for this trial
	tools/mapgen/mapgen.py --no_players=2 --max_hills=1 --min_dimensions=30 --max_dimensions=40 > tmpmap
	# Note that if it doesn't reach save_freq turns, no network is saved.
	# So, repeat until it works:
	while [ ! -e "dqnbot/saved_networks/$FILENAME" ] && [ $TRIES -lt 5 ]
	do
		echo "python tools/playgame.py \"dqnbot/torch/bin/luajit DQNBot.lua -resume -save_freq 100 -save_name "$FILENAME" -nturns "$TURNS"\" \"python tools/sample_bots/python/HunterBot.py\" --nolaunch --map_file tmpmap --log_dir game_logs --turns $TURNS -e"
		python tools/playgame.py "dqnbot/torch/bin/luajit DQNBot.lua -resume -save_freq 100 -save_name "$FILENAME" -nturns "$TURNS"" "python tools/sample_bots/python/HunterBot.py" --nolaunch --map_file tmpmap --log_dir game_logs --turns $TURNS -e
		let TRIES=TRIES+1
	done

	# if still unsuccessful, just quit
	if [ ! -e "dqnbot/saved_networks/$FILENAME" ]
	then
		break;
	fi
done

echo "-DONE-"