 Error termination in NtrErr:
 ntran open failure returned to fopen.
Error: segmentation violation
   rax 0000000000000000, rbx 00000000058e2d00, rcx ffffffffffffffff
   rdx 0000000000002d8b, rsp 00007ffcd10d8ff8, rbp 00007ffcd10d9030
   rsi 000000000000000b, rdi 0000000000002d8b, r8  00002af921d46f80
   r9  00002af921d46f80, r10 00007ffcd10d8a60, r11 0000000000000202
   r12 0000000000000000, r13 00007ffcd10d9940, r14 00007ffcd10da200
   r15 00007ffcd10d9948
addr2line: 'formchk': No such file
  /lib64/libpthread.so.0(+0xf630) [0x2af922247630]
addr2line: 'formchk': No such file
  /lib64/libc.so.6(kill+0x7) [0x2af92278c657]
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
  /lib64/libc.so.6(__libc_start_main+0xf5) [0x2af922778555]
addr2line: 'formchk': No such file
  formchk() [0x40a729]
/var/spool/slurm/job17431399/slurm_script: line 16: 11659 Aborted                 (core dumped) formchk geometry.chk
