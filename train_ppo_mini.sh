#!/bin/bash

mkdir -p logs

stocks=("AEP" "ALGN" "AMAT" "AMD" 
        "AMGN" "AMZN" "ANSS" "ASML" "AVGO" "AZN" "BIIB" "BKNG" 
        "BKR" "CDNS" "CHTR" "CMCSA" "COST" "CPRT" "CSCO" "CSGP" 
        "CSX" "CTAS" "CTSH" "DLTR" "DXCM" "EA" "EBAY" "ENPH" "EXC" 
        "FANG" "FAST" "FTNT" "GILD" "GOOG" "GOOGL" "HON" "IDXX" 
        "ILMN" "INTC" "INTU" "ISRG" "KDP" "KLAC" "LRCX" "LULU" 
        "MAR" "MCHP" "MDLZ" "MELI" "META" "MNST" "MRVL" "MSFT" 
        "MU" "NFLX" "NVDA" "NXPI" "ODFL" "ON" "ORLY" "PANW" "PAYX" 
        "PCAR" "PEP" "QCOM" "REGN" "ROST" "SBUX" "SIRI" "SNPS" 
        "TMUS" "TSLA" "TXN" "VRSK" "VRTX" "WBA" "WBD" "WDAY" "XEL")

counter=0
max_parallel=5

for stock in "${stocks[@]}"
do
    echo "Starting training for $stock"
    python train_ppo_mini.py --stock "$stock" --exp_name "ppo_mini_$stock" > "logs/ppo_100_epochs_20k_steps_64x3_${stock}.txt" 2>&1 &

    ((counter++))

    if (( counter % max_parallel == 0 )); then
        wait
    fi
done

wait
