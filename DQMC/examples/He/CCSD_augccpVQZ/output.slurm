 Error termination in NtrErr:
 ntran open failure returned to fopen.
Error: segmentation violation
   rax 0000000000000000, rbx 00000000058e2d00, rcx ffffffffffffffff
   rdx 0000000000000d05, rsp 00007ffd01f53998, rbp 00007ffd01f539d0
   rsi 000000000000000b, rdi 0000000000000d05, r8  00002ac61198ff80
   r9  00002ac61198ff80, r10 00007ffd01f53420, r11 0000000000000202
   r12 0000000000000000, r13 00007ffd01f542e0, r14 00007ffd01f54ba0
   r15 00007ffd01f542e8
addr2line: 'formchk': No such file
  /lib64/libpthread.so.0(+0xf630) [0x2ac611e91630]
addr2line: 'formchk': No such file
  /lib64/libc.so.6(kill+0x7) [0x2ac6123d6657]
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
  /lib64/libc.so.6(__libc_start_main+0xf5) [0x2ac6123c2555]
addr2line: 'formchk': No such file
  formchk() [0x40a729]
/var/spool/slurm/job17532085/slurm_script: line 16:  3333 Aborted                 (core dumped) formchk geometry.chk
