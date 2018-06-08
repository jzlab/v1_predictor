UNITS=150

for i in 1 2
do
    SAVE_PATH="lnln_hyperopt/$UNITS/run_$i"
    echo "##########################"
    echo "STARTING TRIAL $i"
    echo ""
    echo "SAVE_PATH: $SAVE_PATH"
    echo "##########################"
    mkdir -p $SAVE_PATH

    python lnln_hyperopt.py \
        --learning_rate=0.001 \
        --lnln_units=$UNITS

    mv manualsave/ $SAVE_PATH
done
