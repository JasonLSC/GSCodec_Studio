RD_LAMBDA_LIST="0.0025 0.005 0.01 0.02 0.04"

# TT -> (0,1)
# # 检查脚本是否存在且可执行
if [ ! -x "benchmarks/compression/final_exp/mcmc_tt_sim.sh" ]; then
    echo "Error: mcmc_tt_sim.sh not found or not executable"
    exit 1
fi

for RD_LAMBDA in ${RD_LAMBDA_LIST}; do
    echo "Processing RD_LAMBDA = ${RD_LAMBDA}"
    benchmarks/compression/final_exp/mcmc_tt_sim.sh "${RD_LAMBDA}"
    
    # 检查上一个命令是否成功执行
    if [ $? -ne 0 ]; then
        echo "Error occurred with RD_LAMBDA = ${RD_LAMBDA}"
        # 根据需要决定是继续还是退出
        # exit 1
    fi
done

# # MIP -> (2,3,4,5)
# # 检查脚本是否存在且可执行
# if [ ! -x "benchmarks/compression/final_exp/mcmc_mip_sim.sh" ]; then
#     echo "Error: mcmc_mip_sim.sh not found or not executable"
#     exit 1
# fi

# for RD_LAMBDA in ${RD_LAMBDA_LIST}; do
#     echo "Processing RD_LAMBDA = ${RD_LAMBDA}"
#     benchmarks/compression/final_exp/mcmc_mip_sim.sh "${RD_LAMBDA}"
    
#     # 检查上一个命令是否成功执行
#     if [ $? -ne 0 ]; then
#         echo "Error occurred with RD_LAMBDA = ${RD_LAMBDA}"
#         # 根据需要决定是继续还是退出
#         # exit 1
#     fi
# done

# DB -> (6,7)
# 检查脚本是否存在且可执行
# if [ ! -x "benchmarks/compression/final_exp/mcmc_db_sim.sh" ]; then
#     echo "Error: mcmc_db_sim.sh not found or not executable"
#     exit 1
# fi

# for RD_LAMBDA in ${RD_LAMBDA_LIST}; do
#     echo "Processing RD_LAMBDA = ${RD_LAMBDA}"
#     benchmarks/compression/final_exp/mcmc_db_sim.sh "${RD_LAMBDA}"
    
#     # 检查上一个命令是否成功执行
#     if [ $? -ne 0 ]; then
#         echo "Error occurred with RD_LAMBDA = ${RD_LAMBDA}"
#         # 根据需要决定是继续还是退出
#         # exit 1
#     fi
# done