// SPDX-License-Identifier: Apache-2.0
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <math.h>
#include <type_traits>
using bf16=hip_bfloat16;
// One CTA per (query, head, split). Only selected keys enter QK and PV.
// Tile-local scores never leave LDS. Split states merge with log-sum-exp.
template<bool FAST>
__global__ void sparse(const bf16* q,const bf16* k,const bf16* v,const int* idx,
 float* tmp,bf16* out,int n,int h,int top,int splits,float scale,
 long pqs0,long pqs1,long pqs2,long pks0,long pks1,long pks2,
 long pvs0,long pvs1,long pvs2,long pis0,long pis1) {
 using offset=std::conditional_t<FAST,int,long>;
 offset qs0=FAST ? h*192 : pqs0;
 offset qs1=FAST ? 192 : pqs1;
 offset qs2=FAST ? 1 : pqs2;
 offset ks0=FAST ? n*192 : pks0;
 offset ks1=FAST ? 192 : pks1;
 offset ks2=FAST ? 1 : pks2;
 offset vs0=FAST ? n*128 : pvs0;
 offset vs1=FAST ? 128 : pvs1;
 offset vs2=FAST ? 1 : pvs2;
 offset is0=FAST ? top : pis0;
 offset is1=FAST ? 1 : pis1;

 int row=blockIdx.x,head=blockIdx.y,split=blockIdx.z,t=threadIdx.x;
 int lane=t%64,wave=t/64;
 __shared__ float scores[64],weights[64],partial[512],alpha,denom,maximum;
 __shared__ int positions[64];
 int dlane=lane&31;
 float q0=float(q[row*qs0+head*qs1+(dlane+0)*qs2]);
 float q1=float(q[row*qs0+head*qs1+(dlane+32)*qs2]);
 float q2=float(q[row*qs0+head*qs1+(dlane+64)*qs2]);
 float q3=float(q[row*qs0+head*qs1+(dlane+96)*qs2]);
 float q4=float(q[row*qs0+head*qs1+(dlane+128)*qs2]);
 float q5=float(q[row*qs0+head*qs1+(dlane+160)*qs2]);
 float acc0=0,acc1=0;
 if(t==0){maximum=-INFINITY;denom=0;} __syncthreads();
 int chunk=(top+splits-1)/splits,start=split*chunk,end=min(top,start+chunk);
 for(int base=start;base<end;base+=64){
   if(t<64){int p=base+t; int j=p<end?idx[row*is0+p*is1]:-1;
     positions[t]=(j>=0&&j<n&&j<=row)?j:-1;}
   __syncthreads();
   for(int s=t/32;s<64;s+=8){
     int j=positions[s];float dot=0;
     if(j>=0) dot=q0*float(k[head*ks0+j*ks1+(dlane+0)*ks2])+q1*float(k[head*ks0+j*ks1+(dlane+32)*ks2])+q2*float(k[head*ks0+j*ks1+(dlane+64)*ks2])+q3*float(k[head*ks0+j*ks1+(dlane+96)*ks2])+q4*float(k[head*ks0+j*ks1+(dlane+128)*ks2])+q5*float(k[head*ks0+j*ks1+(dlane+160)*ks2]);
     for(int d=16;d>0;d/=2)dot+=__shfl_down(dot,d,32);
     if(dlane==0)scores[s]=j>=0?dot*scale:-INFINITY;
   } __syncthreads();
   if(wave==0){
     float m=scores[lane];
     for(int d=32;d;d/=2)m=fmaxf(m,__shfl_xor(m,d,64));
     if(lane==0){m=fmaxf(maximum,m);alpha=isfinite(maximum)?expf(maximum-m):0;maximum=m;}
   } __syncthreads();
   if(wave==0){
     float weight=positions[lane]>=0?expf(scores[lane]-maximum):0;
     weights[lane]=weight;
     for(int d=32;d;d/=2)weight+=__shfl_xor(weight,d,64);
     if(lane==0)denom=denom*alpha+weight;
   } __syncthreads();
   acc0*=alpha;acc1*=alpha;
   for(int s=wave;s<64;s+=4){int j=positions[s];if(j>=0){float w=weights[s];
     acc0+=w*float(v[head*vs0+j*vs1+lane*vs2]);
     acc1+=w*float(v[head*vs0+j*vs1+(lane+64)*vs2]);
   }} __syncthreads();
 }
 partial[wave*128+lane]=acc0;partial[wave*128+lane+64]=acc1;
 __syncthreads();
 float acc=t<128?partial[t]+partial[t+128]+partial[t+256]+partial[t+384]:0;
 if(splits==1){if(t<128)out[(long(row)*h+head)*128+t]=bf16(denom>0?acc/denom:0);return;}
 long off=((long(row)*h+head)*splits+split)*130;
 if(t<128)tmp[off+t]=acc;
 if(t==0){tmp[off+128]=maximum;tmp[off+129]=denom;}
}
__global__ void merge(const float* tmp,bf16* out,int h,int splits){
 int row=blockIdx.x,head=blockIdx.y,t=threadIdx.x;
 long off=(long(row)*h+head)*splits*130;
 float m=-INFINITY;for(int s=0;s<splits;s++)m=fmaxf(m,tmp[off+s*130+128]);
 float sum=0,val=0;
 for(int s=0;s<splits;s++){float l=tmp[off+s*130+129];
   float a=l>0?expf(tmp[off+s*130+128]-m):0;
   sum+=l*a;val+=tmp[off+s*130+t]*a;}
 out[(long(row)*h+head)*128+t]=bf16(sum>0?val/sum:0);
}

// Fixed top-k<=2048 alternative: row-local LDS only, no global score/P tensor.
// One stable row softmax; compared separately against streaming online path.
__global__ void sparse_row(const bf16* q,const bf16* k,const bf16* v,const int* idx,bf16* out,
 int n,int h,int top,float scale,long qs0,long qs1,long qs2,long ks0,long ks1,long ks2,
 long vs0,long vs1,long vs2,long is0,long is1){
 int row=blockIdx.x,head=blockIdx.y,t=threadIdx.x,lane=t%64,wave=t/64;
 __shared__ float score[2048],part[512],maximum,denom;
 __shared__ int pos[2048];
 for(int s=t;s<top;s+=256){int j=idx[row*is0+s*is1];pos[s]=j>=0&&j<n&&j<=row?j:-1;}
 __syncthreads();
 float q0=float(q[row*qs0+head*qs1+lane*qs2]);
 float q1=float(q[row*qs0+head*qs1+(lane+64)*qs2]);
 float q2=float(q[row*qs0+head*qs1+(lane+128)*qs2]);
 for(int s=wave;s<top;s+=4){int j=pos[s];float dot=0;
  if(j>=0)dot=q0*float(k[head*ks0+j*ks1+lane*ks2])+q1*float(k[head*ks0+j*ks1+(lane+64)*ks2])+q2*float(k[head*ks0+j*ks1+(lane+128)*ks2]);
  for(int d=32;d;d/=2)dot+=__shfl_down(dot,d,64);
  if(lane==0)score[s]=j>=0?dot*scale:-INFINITY;
 }__syncthreads();
 float m=-INFINITY;
 for(int s=t;s<top;s+=256)m=fmaxf(m,score[s]);
 for(int d=32;d;d/=2)m=fmaxf(m,__shfl_xor(m,d,64));
 if(lane==0)part[wave]=m;__syncthreads();
 if(t==0)maximum=fmaxf(fmaxf(part[0],part[1]),fmaxf(part[2],part[3]));__syncthreads();
 float sum=0;
 for(int s=t;s<top;s+=256){float p=pos[s]>=0?expf(score[s]-maximum):0;score[s]=p;sum+=p;}
 for(int d=32;d;d/=2)sum+=__shfl_xor(sum,d,64);
 if(lane==0)part[wave]=sum;__syncthreads();
 if(t==0)denom=part[0]+part[1]+part[2]+part[3];__syncthreads();
 float a=0,b=0;
 for(int s=wave;s<top;s+=4){int j=pos[s];if(j>=0){float w=score[s];
 a+=w*float(v[head*vs0+j*vs1+lane*vs2]);b+=w*float(v[head*vs0+j*vs1+(lane+64)*vs2]);}}
 part[wave*128+lane]=a;part[wave*128+lane+64]=b;__syncthreads();
 if(t<128){float val=part[t]+part[t+128]+part[t+256]+part[t+384];
 out[(long(row)*h+head)*128+t]=bf16(denom>0?val/denom:0);}
}

extern "C" int kimi_fused_sparse(const void*q,const void*k,const void*v,const int*i,
 float*tmp,void*out,int n,int h,int top,int splits,float scale,
 const long* st,void* stream){
 hipStream_t s=(hipStream_t)stream;
 if(splits==0){
 hipLaunchKernelGGL(sparse_row,dim3(n,h),dim3(256),0,s,(const bf16*)q,(const bf16*)k,(const bf16*)v,i,(bf16*)out,n,h,top,scale,st[0],st[1],st[2],st[3],st[4],st[5],st[6],st[7],st[8],st[9],st[10]);return int(hipGetLastError());}
 if(long(n)*h*192<2147483647 && long(n)*top<2147483647 && st[0]==h*192 && st[1]==192 && st[2]==1 && st[3]==n*192 && st[4]==192 && st[5]==1 && st[6]==n*128 && st[7]==128 && st[8]==1 && st[9]==top && st[10]==1) {
 hipLaunchKernelGGL(sparse<true>,dim3(n,h,splits),dim3(256),0,s,
 (const bf16*)q,(const bf16*)k,(const bf16*)v,i,tmp,(bf16*)out,n,h,top,splits,scale,
 st[0],st[1],st[2],st[3],st[4],st[5],st[6],st[7],st[8],st[9],st[10]);
 } else {
 hipLaunchKernelGGL(sparse<false>,dim3(n,h,splits),dim3(256),0,s,
 (const bf16*)q,(const bf16*)k,(const bf16*)v,i,tmp,(bf16*)out,n,h,top,splits,scale,
 st[0],st[1],st[2],st[3],st[4],st[5],st[6],st[7],st[8],st[9],st[10]);
 }
 auto e=hipGetLastError();if(e!=hipSuccess)return int(e);
 if(splits==1)return int(hipSuccess);
 hipLaunchKernelGGL(merge,dim3(n,h),dim3(128),0,s,tmp,(bf16*)out,h,splits);
 return int(hipGetLastError());
}
