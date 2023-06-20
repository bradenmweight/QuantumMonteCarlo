/var/spool/slurm/job17429886/slurm_script: line 16: 29114 Aborted                 (core dumped) formchk geometry.chk
 Error termination in NtrErr:
 ntran open failure returned to fopen.
Error: segmentation violation
   rax 0000000000000000, rbx 00000000058e2d00, rcx ffffffffffffffff
   rdx 00000000000073bf, rsp 00007ffd33ba1b48, rbp 00007ffd33ba1b80
   rsi 000000000000000b, rdi 00000000000073bf, r8  00002b5693c11f80
   r9  00002b5693c11f80, r10 00007ffd33ba15a0, r11 0000000000000206
   r12 0000000000000000, r13 00007ffd33ba2490, r14 00007ffd33ba2d50
   r15 00007ffd33ba2498
addr2line: 'formchk': No such file
  /lib64/libpthread.so.0(+0xf630) [0x2b5694112630]
addr2line: 'formchk': No such file
  /lib64/libc.so.6(kill+0x7) [0x2b5694657657]
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
  /lib64/libc.so.6(__libc_start_main+0xf5) [0x2b5694643555]
addr2line: 'formchk': No such file
  formchk() [0x40a729]
/var/spool/slurm/job17429896/slurm_script: line 16: 29631 Aborted                 (core dumped) formchk geometry.chk
