 Error termination in NtrErr:
 ntran open failure returned to fopen.
Error: segmentation violation
   rax 0000000000000000, rbx 00000000058e2d00, rcx ffffffffffffffff
   rdx 0000000000007e72, rsp 00007ffd40acdfe8, rbp 00007ffd40ace020
   rsi 000000000000000b, rdi 0000000000007e72, r8  00002acc309baf80
   r9  00002acc309baf80, r10 00007ffd40acda60, r11 0000000000000206
   r12 0000000000000000, r13 00007ffd40ace930, r14 00007ffd40acf1f0
   r15 00007ffd40ace938
addr2line: 'formchk': No such file
  /lib64/libpthread.so.0(+0xf630) [0x2acc30ebc630]
addr2line: 'formchk': No such file
  /lib64/libc.so.6(kill+0x7) [0x2acc31401657]
addr2line: 'formchk': No such file
  formchk() [0x461f15]
addr2line: 'formchk': No such file
  formchk() [0x4334b9]
addr2line: 'formchk': No such file
  formchk() [0x480882]
addr2line: 'formchk': No such file
  formchk() [0x45406e]
addr2line: 'formchk': No such file
  formchk() [0x4108b7]
addr2line: 'formchk': No such file
  formchk() [0x40ffe4]
addr2line: 'formchk': No such file
  /lib64/libc.so.6(__libc_start_main+0xf5) [0x2acc313ed555]
addr2line: 'formchk': No such file
  formchk() [0x40a729]
/var/spool/slurm/job17532013/slurm_script: line 16: 32370 Aborted                 (core dumped) formchk geometry.chk
