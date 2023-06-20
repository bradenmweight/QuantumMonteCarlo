 Error termination in NtrErr:
 ntran open failure returned to fopen.
Error: segmentation violation
   rax 0000000000000000, rbx 00000000058e2d00, rcx ffffffffffffffff
   rdx 00000000000074b0, rsp 00007ffd74619218, rbp 00007ffd74619250
   rsi 000000000000000b, rdi 00000000000074b0, r8  00002af1183c7f80
   r9  00002af1183c7f80, r10 00007ffd74618ca0, r11 0000000000000206
   r12 0000000000000000, r13 00007ffd74619b60, r14 00007ffd7461a420
   r15 00007ffd74619b68
addr2line: 'formchk': No such file
  /lib64/libpthread.so.0(+0xf630) [0x2af1188c8630]
addr2line: 'formchk': No such file
  /lib64/libc.so.6(kill+0x7) [0x2af118e0d657]
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
  /lib64/libc.so.6(__libc_start_main+0xf5) [0x2af118df9555]
addr2line: 'formchk': No such file
  formchk() [0x40a729]
/var/spool/slurm/job17429907/slurm_script: line 16: 29872 Aborted                 (core dumped) formchk geometry.chk
