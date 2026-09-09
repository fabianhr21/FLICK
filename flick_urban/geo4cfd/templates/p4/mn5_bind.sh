#!/bin/bash

case ${OMPI_COMM_WORLD_LOCAL_RANK} in
0)
export CUDA_VISIBLE_DEVICES=0
export OMPI_MCA_btl_openib_if_include=mlx5_0:1
export UCX_NET_DEVICES=mlx5_0:1
numactl --membind=0 "$@"
  ;;
1)
export CUDA_VISIBLE_DEVICES=1
export UCX_NET_DEVICES=mlx5_1:1
export OMPI_MCA_btl_openib_if_include=mlx5_1:1
numactl --membind=1 "$@"
  ;;
2)
export CUDA_VISIBLE_DEVICES=2
export UCX_NET_DEVICES=mlx5_4:1
export OMPI_MCA_btl_openib_if_include=mlx5_4:1
numactl --membind=2 "$@"
  ;;
3)
export CUDA_VISIBLE_DEVICES=3
export UCX_NET_DEVICES=mlx5_5:1
export OMPI_MCA_btl_openib_if_include=mlx5_5:1
numactl --membind=3 "$@"
  ;;
esac

