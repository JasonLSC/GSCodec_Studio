#!/bin/bash

RD_LAMBDA_LIST="0.002 0.004 0.0075 0.006 0.008"

# 定义TT执行函数
run_tt() {
    if [ ! -x "benchmarks/compression/final_exp/mcmc_tt_sim.sh" ]; then
        echo "Error: mcmc_tt_sim.sh not found or not executable"
        return 1
    fi

    for RD_LAMBDA in ${RD_LAMBDA_LIST}; do
        echo "Processing TT with RD_LAMBDA = ${RD_LAMBDA}"
        benchmarks/compression/final_exp/mcmc_tt_sim.sh "${RD_LAMBDA}"
        
        if [ $? -ne 0 ]; then
            echo "Error occurred in TT with RD_LAMBDA = ${RD_LAMBDA}"
            return 1
        fi
    done
}

# 定义MIP执行函数
run_mip() {
    if [ ! -x "benchmarks/compression/final_exp/mcmc_mip_sim.sh" ]; then
        echo "Error: mcmc_mip_sim.sh not found or not executable"
        return 1
    fi

    for RD_LAMBDA in ${RD_LAMBDA_LIST}; do
        echo "Processing MIP with RD_LAMBDA = ${RD_LAMBDA}"
        benchmarks/compression/final_exp/mcmc_mip_sim.sh "${RD_LAMBDA}"
        
        if [ $? -ne 0 ]; then
            echo "Error occurred in MIP with RD_LAMBDA = ${RD_LAMBDA}"
            return 1
        fi
    done
}

# 定义DB执行函数
run_db() {
    if [ ! -x "benchmarks/compression/final_exp/mcmc_db_sim.sh" ]; then
        echo "Error: mcmc_db_sim.sh not found or not executable"
        return 1
    fi

    for RD_LAMBDA in ${RD_LAMBDA_LIST}; do
        echo "Processing DB with RD_LAMBDA = ${RD_LAMBDA}"
        benchmarks/compression/final_exp/mcmc_db_sim.sh "${RD_LAMBDA}"
        
        if [ $? -ne 0 ]; then
            echo "Error occurred in DB with RD_LAMBDA = ${RD_LAMBDA}"
            return 1
        fi
    done
}

# 并行执行三个函数
run_tt &
tt_pid=$!

# run_mip &
# mip_pid=$!

# run_db &
# db_pid=$!

# # 等待所有进程完成
wait $tt_pid
tt_status=$?
# wait $mip_pid
# mip_status=$?
# wait $db_pid
# db_status=$?

# # 检查是否所有进程都成功完成
if [ $tt_status -ne 0 ] || [ $mip_status -ne 0 ] || [ $db_status -ne 0 ]; then
    echo "One or more processes failed"
    exit 1
fi

echo "All processes completed successfully"