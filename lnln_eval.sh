#! /bin/sh

# Decided on from hyperparameter sweep
UNITS=250

for i in 1 2 3 4 5 6
do

    for fileindex in 7 9
    do
        SAVE_PATH="lnln_eval/kohn_$fileindex"
        echo "##########################"
        echo "STARTING TRIAL $i"
        echo ""
        echo "SAVE_PATH: $SAVE_PATH"
        echo "##########################"
        mkdir -p $SAVE_PATH

        python lnln_eval.py \
            --learning_rate=0.001 \
            --lnln_units=$UNITS \
            --fileindex=$fileindex \
            --max_steps=60000 \
            --savetraining=False \

        mv manualsave/*.npy $SAVE_PATH/
    done

done
