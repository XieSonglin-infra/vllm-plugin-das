// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-FileCopyrightText: Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
// Design adapted from vLLM bec08b70193cef26c1b9a6bd39c44bf7f4e031e6,
// vllm/models/kimi_k3/amd/ops/flydsl_sparse_mla_gfx950.py.
// That file attributes the MIT-licensed ROCm/aiter source at
// a61e02c26a3915011e85b103574c25d457890825; that AITER revision is not
// independently verified here. This HCU BF16 implementation is a rewrite:
// no gfx950 scaled MFMA, FP8 codec/record, or Q residual quantization is used.
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>
#include <math.h>
using bf16=hip_bfloat16;
using u4=unsigned short __attribute__((ext_vector_type(4)));
using f4=float __attribute__((ext_vector_type(4)));
static __device__ __forceinline__ f4 mmac(u4 a,u4 b,f4 c){
#if defined(__gfx938__)
 return __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a,b,c,true,true);
#elif defined(__HIP_DEVICE_COMPILE__)
#error "This standalone migration requires validated gfx938 MMAC layout"
#else
 return c;
#endif
}
// Core indices are PHYSICAL slots. Causality belongs to the caller's indexer.
template<int TILE>
__global__ __launch_bounds__(256) void migrated(
 const bf16* q,const bf16* cache,const int* idx,const int* lengths,bf16* out,
 int n,int h,int rows,int top,int page,
 int64_t qs0,int64_t qs1,int64_t qs2,int64_t bs,int64_t ts,int64_t ds,
 int64_t is0,int64_t is1,int64_t ls,float scale){
 int row=n-1-int(blockIdx.x),t=threadIdx.x,lane=t&63,wave=t>>6;
 int input_head=lane&15,k_lane=(lane>>4)*4;
 __shared__ unsigned short kv[TILE][576];
 __shared__ unsigned short p[16][TILE];
 __shared__ float score[16][TILE],maxima[16],denom[16],alpha[16];
 __shared__ int positions[TILE],lasts[4],last;
 __shared__ int64_t offsets[TILE];
 int bound=lengths?max(0,min(top,lengths[int64_t(row)*ls])):top;
 int latest=0;
 for(int s=t;s<bound;s+=256){int j=idx[int64_t(row)*is0+int64_t(s)*is1];
  if(j>=0&&j<rows)latest=s+1;}
 for(int d=32;d;d>>=1)latest=max(latest,__shfl_down(latest,d,64));
 if(lane==0)lasts[wave]=latest;
 if(t<16){maxima[t]=-INFINITY;denom[t]=0;}
 __syncthreads();
 if(t==0)last=max(max(lasts[0],lasts[1]),max(lasts[2],lasts[3]));
 __syncthreads();
 if(last==0){for(int i=t;i<h*512;i+=256)out[int64_t(row)*h*512+i]=bf16(0);return;}
 f4 accum[8];
 #pragma unroll
 for(int b=0;b<8;b++)accum[b]=f4{0,0,0,0};
 for(int base=0;base<last;base+=TILE){
  if(t<TILE){int j=base+t<bound?idx[int64_t(row)*is0+int64_t(base+t)*is1]:-1;
   bool valid=j>=0&&j<rows;positions[t]=valid?j:-1;
   offsets[t]=valid?(int64_t(j/page)*bs+int64_t(j%page)*ts):0;}
  __syncthreads();
  for(int i=t;i<TILE*576;i+=256){int token=i/576,d=i%576;
   kv[token][d]=positions[token]>=0?reinterpret_cast<const unsigned short*>(cache)[offsets[token]+int64_t(d)*ds]:0;}
  __syncthreads();
  if(wave<TILE/16){
   f4 dots{0,0,0,0};
   // QK: one wave computes all <=16 heads against 16 gathered KV slots.
   #pragma unroll 1
   for(int d=0;d<576;d+=16){
    u4 a,b;
    #pragma unroll
    for(int r=0;r<4;r++){
     a[r]=input_head<h?reinterpret_cast<const unsigned short*>(q)[int64_t(row)*qs0+int64_t(input_head)*qs1+int64_t(d+k_lane+r)*qs2]:0;
     b[r]=kv[wave*16+(lane&15)][d+k_lane+r];
    }
    dots=mmac(a,b,dots);
   }
   #pragma unroll
   for(int r=0;r<4;r++){int head=(lane>>4)*4+r,token=wave*16+(lane&15);
    score[head][token]=(head<h&&positions[token]>=0)?dots[r]*scale:-INFINITY;}
  }
  __syncthreads();
  int head=t>>4,col=t&15;
  float m=-INFINITY;
  #pragma unroll
  for(int x=0;x<TILE;x+=16)m=fmaxf(m,score[head][x+col]);
  for(int d=8;d;d>>=1)m=fmaxf(m,__shfl_xor(m,d,16));
  m=fmaxf(m,maxima[head]);
  float a=isfinite(maxima[head])?expf(maxima[head]-m):0;
  float l=0;
  #pragma unroll
  for(int x=0;x<TILE;x+=16){int token=x+col;
   float w=(head<h&&positions[token]>=0)?expf(score[head][token]-m):0;
   bf16 rounded(w);p[head][token]=*reinterpret_cast<unsigned short*>(&rounded);
   l+=w;
  }
  for(int d=8;d;d>>=1)l+=__shfl_xor(l,d,16);
  if(col==0){alpha[head]=a;maxima[head]=m;denom[head]=denom[head]*a+l;}
  __syncthreads();
  // PV: four waves own disjoint 128-dimensional latent output stripes.
  #pragma unroll
  for(int b=0;b<8;b++){
   #pragma unroll
   for(int r=0;r<4;r++)accum[b][r]*=alpha[(lane>>4)*4+r];
   #pragma unroll
   for(int kk=0;kk<TILE;kk+=16){
    u4 a,bv;
    #pragma unroll
    for(int r=0;r<4;r++){
     a[r]=p[input_head][kk+k_lane+r];
     bv[r]=kv[kk+k_lane+r][wave*128+b*16+(lane&15)];
    }
    accum[b]=mmac(a,bv,accum[b]);
   }
  }
  __syncthreads();
 }
 #pragma unroll
 for(int b=0;b<8;b++){
  #pragma unroll
  for(int r=0;r<4;r++){int head=(lane>>4)*4+r;
   if(head<h){float l=denom[head];out[(int64_t(row)*h+head)*512+wave*128+b*16+(lane&15)]=bf16(l>0?accum[b][r]/l:0);}}
 }
}
extern "C" int migrated_sparse_mla(
 const void* q,const void* cache,const int* idx,const int* lengths,void* out,
 int n,int h,int rows,int top,int page,int tile,const int64_t* st,float scale,void* stream){
 if(tile==32){
 hipLaunchKernelGGL(migrated<32>,dim3(n),dim3(256),0,(hipStream_t)stream,
 (const bf16*)q,(const bf16*)cache,idx,lengths,(bf16*)out,n,h,rows,top,page,
 st[0],st[1],st[2],st[3],st[4],st[5],st[6],st[7],st[8],scale);
 }else if(tile==16){
 hipLaunchKernelGGL(migrated<16>,dim3(n),dim3(256),0,(hipStream_t)stream,
 (const bf16*)q,(const bf16*)cache,idx,lengths,(bf16*)out,n,h,rows,top,page,
 st[0],st[1],st[2],st[3],st[4],st[5],st[6],st[7],st[8],scale);
 }else{return int(hipErrorInvalidValue);}
 return int(hipGetLastError());
}
__global__ void layout_probe(const unsigned short*q,const unsigned short*k,float*out){
 int lane=threadIdx.x;u4 a,b;
 for(int r=0;r<4;r++){a[r]=q[(lane%16)*16+(lane/16)*4+r];b[r]=k[(lane%16)*16+(lane/16)*4+r];}
 f4 c=mmac(a,b,f4{0,0,0,0});
 for(int r=0;r<4;r++)out[((lane/16)*4+r)*16+lane%16]=c[r];
}
extern "C" int migrated_layout_probe(const void*q,const void*k,void*out,void*stream){
 hipLaunchKernelGGL(layout_probe,dim3(1),dim3(64),0,(hipStream_t)stream,(const unsigned short*)q,(const unsigned short*)k,(float*)out);
 return int(hipGetLastError());
}
extern "C" int migrated_device_info(int* result){hipDeviceProp_t p;int d;hipGetDevice(&d);auto e=hipGetDeviceProperties(&p,d);
 if(e==hipSuccess){result[0]=p.sharedMemPerBlock;result[1]=p.warpSize;result[2]=p.multiProcessorCount;}return int(e);}
