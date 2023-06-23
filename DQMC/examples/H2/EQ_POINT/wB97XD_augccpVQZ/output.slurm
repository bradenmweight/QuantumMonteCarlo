 Error termination in NtrErr:
 ntran open failure returned to fopen.
Error: segmentation violation
   rax 0000000000000000, rbx 00000000058e2d00, rcx ffffffffffffffff
   rdx 00000000000035fe, rsp 00007ffce8b72088, rbp 00007ffce8b720c0
   rsi 000000000000000b, rdi 00000000000035fe, r8  00002aced360bf80
   r9  00002aced360bf80, r10 00007ffce8b71ae0, r11 0000000000000206
   r12 0000000000000000, r13 00007ffce8b729d0, r14 00007ffce8b73290
   r15 00007ffce8b729d8
addr2line: 'formchk': No such file
  /lib64/libpthread.so.0(+0xf630) [0x2aced3b0d630]
addr2line: 'formchk': No such file
  /lib64/libc.so.6(kill+0x7) [0x2aced4052657]
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
  /lib64/libc.so.6(__libc_start_main+0xf5) [0x2aced403e555]
addr2line: 'formchk': No such file
  formchk() [0x40a729]
/var/spool/slurm/job17541196/slurm_script: line 16: 13822 Aborted                 (core dumped) formchk geometry.chk
