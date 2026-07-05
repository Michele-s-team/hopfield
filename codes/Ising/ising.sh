#!/bin/bash

# =============================================================================
# run_ising_parallel.sh
# Launches multiple simulations in parallel for different system sizes
# =============================================================================

PROG="main.o"
LOG_DIR="logs"
mkdir -p $LOG_DIR

# -- Compilation --------------------------------------------------
echo "Compilation..."
rm -f $PROG

g++ main.cpp src/*.cpp \
    -I ./include \
    -I /usr/include/gsl \
    -lgsl -lgslcblas -lm \
    -O3 -Wno-deprecated -DHAVE_INLINE \
    -o $PROG

if [ $? -ne 0 ]; then
    echo "Compilation error!"
    exit 1
fi
echo "Compilation successful"
echo ""

# -- List of N (number of spins) --------------------------------
# N = L x L
# Modify according to your needs
N_LIST=(100 400 900 1600 2500 3600 4900 6400 8100 10000 12544 15625 22500)

# Version with corresponding L values
# L_LIST=(10 20 30 40 50 60 70 80 90 100 112 125 150)
# N_LIST=()
# for L in "${L_LIST[@]}"; do
#     N_LIST+=($((L * L)))
# done

# -- Launch simulations in parallel ------------------------------
echo "Launching simulations in parallel..."
echo "Number of simulations: ${#N_LIST[@]}"
echo "N values: ${N_LIST[@]}"
echo ""

START_TIME=$(date +%s)

for N in "${N_LIST[@]}"; do
    LOG_FILE="$LOG_DIR/simulation_N${N}_$(date +%Y%m%d_%H%M%S).log"
    echo "Launching N=$N"
    
    # Launch in background
    ./$PROG $N > "$LOG_FILE" 2>&1 &
    
    # Small pause to avoid write conflicts at startup
    sleep 0.5
done

echo ""
echo "All simulations have been launched in parallel."
echo "Number of active processes: $(jobs -r | wc -l)"
echo ""
echo "Waiting for all processes to finish..."

# Wait for all background processes to complete
wait

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "================================================================"
echo "All simulations are complete!"
echo "Total execution time: $((DURATION / 3600))h $(((DURATION % 3600) / 60))m $((DURATION % 60))s"
echo "================================================================"

# Display a summary
echo ""
echo "Log summary:"
ls -lh $LOG_DIR/