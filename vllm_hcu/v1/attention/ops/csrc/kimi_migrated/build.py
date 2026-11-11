# SPDX-License-Identifier: Apache-2.0
"""Incremental gfx938 migration build; all generated files stay in task output."""
import hashlib,json,os,subprocess
from pathlib import Path
base=Path(__file__).resolve().parent
out=Path(os.environ.get("VLLM_HCU_MIGRATED_BUILD_DIR",
 "/home/xiesl/inference/send_time/task_dir/speedup_kimi31pro/migrated_sparse_mla/build"))
out.mkdir(parents=True,exist_ok=True)
compiler="/opt/dtk/bin/hipcc"
cmd=[compiler,"-O3","-std=c++17","-fPIC","-shared","--offload-arch=gfx938",
 "-save-temps",str(base/"migrated_sparse_mla.hip"),"-o",str(out/"migrated_sparse_mla.so")]
inputs=[p for p in base.rglob("*") if p.is_file() and p.suffix in (".hip",".hpp",".h",".py")]
sdk=[Path("/opt/dtk/hip/include/hip")/name for name in ("hip_runtime.h","hip_runtime_api.h","hip_bfloat16.h")]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs+sdk if p.exists()}
manifest={"sources":hashes,"compiler":subprocess.check_output([compiler,"--version"],text=True),
          "arch":"gfx938","command":cmd,"numerics":"BF16 operands, FP32 MMAC/softmax/PV accumulation"}
mp=out/"manifest.json"
if not mp.exists() or json.loads(mp.read_text())!=manifest or not (out/"migrated_sparse_mla.so").exists():
 temporary=out/("."+str(os.getpid())+".tmp.so")
 actual=cmd[:-1]+[str(temporary)]
 r=subprocess.run(actual,cwd=out,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 log=out/("compile_"+hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:12]+".log")
 log.write_text(r.stdout)
 (out/"compile.log").write_text(r.stdout)
 (out/"last_command.json").write_text(json.dumps(actual,indent=2))
 r.check_returncode()
 os.replace(temporary,out/"migrated_sparse_mla.so")
 mp.write_text(json.dumps(manifest,indent=2))
print(out/"migrated_sparse_mla.so")
