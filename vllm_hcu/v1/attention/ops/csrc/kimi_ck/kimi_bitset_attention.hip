// SPDX-License-Identifier: Apache-2.0
// Uses Hygon CK Tile MIT headers; local overlay preserves original notices.
#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue.hpp"
using namespace ck_tile;
struct KimiBitMask:GenericAttentionMask<true,false>{
 using Base=GenericAttentionMask<true,false>;
 int ny,nx;
 CK_TILE_HOST_DEVICE KimiBitMask(int y,int x,int sink,int n,int m):Base(y,x,sink,n,m),ny(n),nx(m){}
 CK_TILE_HOST_DEVICE KimiBitMask(int n,int m):Base(n,m),ny(n),nx(m){}
 const uint32_t* kimi_bits=nullptr;
 static constexpr const char* name="kimi_bitset";
 CK_TILE_DEVICE bool IsOutOfBound(index_t row,index_t col) const {
  return row>=ny||col>=nx||col>row||!(kimi_bits[row*((nx+31)/32)+col/32]&(1u<<(col%32)));
 }
 template<index_t M,index_t N>
 CK_TILE_DEVICE bool IsEdgeTile(index_t,index_t,number<M>,number<N>) const{return true;}
 template<index_t M,index_t N>
 CK_TILE_DEVICE bool IsTileActive(index_t row,index_t col,number<M>,number<N>) const{
  uint32_t active=0;int words=(nx+31)/32;
  for(int r=row;r<row+M&&r<ny;r++)
   for(int w=col/32;w<(col+N+31)/32&&w<words;w++)active|=kimi_bits[r*words+w];
  return __builtin_amdgcn_readfirstlane(active)!=0;
 }
};
using Shape=TileFmhaShape<sequence<16,64,32,64,64,256>,
 sequence<1,1,1>,sequence<16,64,32>,sequence<1,1,1>,sequence<16,32,64>,true>;
using Traits=TileFmhaTraits<true,true,true,true,BlockAttentionBiasEnum::NO_BIAS,false,false,false,false>;
using Problem=BlockFmhaPipelineProblem<bf16_t,bf16_t,bf16_t,float,float,bf16_t,uint8_t,float,bf16_t,float,bf16_t,Shape,false,KimiBitMask,Traits>;
using Pipeline=BlockFmhaPipelineQRKSVS<Problem>;
using Epilogue=Default2DEpilogue<Default2DEpilogueProblem<float,bf16_t,true,true>>;
using Partitioner=FmhaFwdTilePartitioner_SHB<Shape>;
using Kernel=FmhaFwdKernel<Partitioner,Pipeline,Epilogue>;
__global__ void pack_bits(const int* idx,uint32_t* bits,int n,int top,int is0,int is1){
 int row=blockIdx.x,t=threadIdx.x,words=(n+31)/32;
 for(int w=t;w<words;w+=128)bits[row*words+w]=0;
 __syncthreads();
 for(int s=t;s<top;s+=128){int j=idx[row*is0+s*is1];
  if(j>=0&&j<n&&j<=row)atomicOr(bits+row*words+j/32,1u<<(j%32));}
}
__global__ __launch_bounds__(64) void attention(Kernel::Kargs args){Kernel{}(args);}
extern "C" int kimi_ck_pack(const int* idx,uint32_t* bits,int n,int top,int is0,int is1,void* stream){
 hipLaunchKernelGGL(pack_bits,dim3(n),dim3(128),0,(hipStream_t)stream,idx,bits,n,top,is0,is1);
 return int(hipGetLastError());
}
extern "C" int kimi_ck_attention(const void*q,const void*k,const void*v,const uint32_t*bits,void*out,
 int n,int h,float scale,void* stream){
 Kernel::Kargs args{};
 args.q_ptr=q;args.k_ptr=k;args.v_ptr=v;args.o_ptr=out;
 args.seqlen_q=n;args.seqlen_k=n;args.hdim_q=192;args.hdim_v=128;
 args.num_head_q=h;args.nhead_ratio_qk=1;
#if CK_TILE_FMHA_FWD_FAST_EXP2
 args.scale_s=scale*ck_tile::log2e_v<>;
#else
 args.scale_s=scale;
#endif
 args.stride_q=h*192;args.stride_k=192;args.stride_v=128;args.stride_o=h*128;
 args.nhead_stride_q=192;args.nhead_stride_k=n*192;args.nhead_stride_v=n*128;args.nhead_stride_o=128;
 args.batch_stride_q=n*h*192;args.batch_stride_k=n*h*192;args.batch_stride_v=n*h*128;args.batch_stride_o=n*h*128;
 args.window_size_left=-1;args.window_size_right=0;args.mask_type=GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT;
 args.kimi_bits=bits;
 auto grid=Kernel::GridSize(1,h,n,128);
 hipLaunchKernelGGL(attention,grid,dim3(64),0,(hipStream_t)stream,args);
 return int(hipGetLastError());
}
