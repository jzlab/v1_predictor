

# Decided on from hyperparameter sweep
UNITS=250

for i in 1 2 3 4 5 6 7
do

    for fileindex in 1 3 4 5 6 7 8 9 10
    do
        SAVE_PATH="lnln_eval/kohn_$fileindex/trial_$i"
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

        mv manualsave/ $SAVE_PATH
    done

done
