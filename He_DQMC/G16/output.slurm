 Error termination in NtrErr:
 ntran open failure returned to fopen.
Error: segmentation violation
   rax 0000000000000000, rbx 00000000058e2d00, rcx ffffffffffffffff
   rdx 0000000000002788, rsp 00007fff463d9748, rbp 00007fff463d9780
   rsi 000000000000000b, rdi 0000000000002788, r8  00002b42be24ff80
   r9  00002b42be24ff80, r10 00007fff463d91a0, r11 0000000000000206
   r12 0000000000000000, r13 00007fff463da090, r14 00007fff463da950
   r15 00007fff463da098
addr2line: 'formchk': No such file
  /lib64/libpthread.so.0(+0xf630) [0x2b42be74e630]
addr2line: 'formchk': No such file
  /lib64/libc.so.6(kill+0x7) [0x2b42bec93657]
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
  /lib64/libc.so.6(__libc_start_main+0xf5) [0x2b42bec7f555]
addr2line: 'formchk': No such file
  formchk() [0x40a729]
/var/spool/slurm/job17431386/slurm_script: line 16: 10120 Aborted                 (core dumped) formchk geometry.chk
