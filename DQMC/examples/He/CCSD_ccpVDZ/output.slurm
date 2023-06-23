 Error termination in NtrErr:
 ntran open failure returned to fopen.
Error: segmentation violation
   rax 0000000000000000, rbx 00000000058e2d00, rcx ffffffffffffffff
   rdx 0000000000000b50, rsp 00007ffca1d111d8, rbp 00007ffca1d11210
   rsi 000000000000000b, rdi 0000000000000b50, r8  00002b25b4a73f80
   r9  00002b25b4a73f80, r10 00007ffca1d10c60, r11 0000000000000206
   r12 0000000000000000, r13 00007ffca1d11b20, r14 00007ffca1d123e0
   r15 00007ffca1d11b28
addr2line: 'formchk': No such file
  /lib64/libpthread.so.0(+0xf630) [0x2b25b4f75630]
addr2line: 'formchk': No such file
  /lib64/libc.so.6(kill+0x7) [0x2b25b54ba657]
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
  /lib64/libc.so.6(__libc_start_main+0xf5) [0x2b25b54a6555]
addr2line: 'formchk': No such file
  formchk() [0x40a729]
/var/spool/slurm/job17532082/slurm_script: line 16:  2896 Aborted                 (core dumped) formchk geometry.chk
