 Error termination in NtrErr:
 ntran open failure returned to fopen.
Error: segmentation violation
   rax 0000000000000000, rbx 00000000058e2d00, rcx ffffffffffffffff
   rdx 0000000000002aa0, rsp 00007ffd42465638, rbp 00007ffd42465670
   rsi 000000000000000b, rdi 0000000000002aa0, r8  00002b50b23b2f80
   r9  00002b50b23b2f80, r10 00007ffd424650a0, r11 0000000000000202
   r12 0000000000000000, r13 00007ffd42465f80, r14 00007ffd42466840
   r15 00007ffd42465f88
addr2line: 'formchk': No such file
  /lib64/libpthread.so.0(+0xf630) [0x2b50b28b3630]
addr2line: 'formchk': No such file
  /lib64/libc.so.6(kill+0x7) [0x2b50b2df8657]
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
  /lib64/libc.so.6(__libc_start_main+0xf5) [0x2b50b2de4555]
addr2line: 'formchk': No such file
  formchk() [0x40a729]
/var/spool/slurm/job17431389/slurm_script: line 16: 10912 Aborted                 (core dumped) formchk geometry.chk
