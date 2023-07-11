#this code performs QED-CCSD calculations with complete double excitations
#t = t1_10 + t1_01 + t2_20 + t2_02 + t2_11 + t2_21 + t2_12 + t2_22
import numpy as np
import math
from pkg_resources import parse_version
from numba import jit
import time
import scipy.linalg as la
import sys
import subprocess as sp

import psi4


#QED-CCSD equations derived with Wick (https://github.com/awhite862/wick)
def ccsd_t2_20(f_so, g_so, dip, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nvir = t2_20.shape[0]
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]
    e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir.reshape(-1, 1, 1) - eps_vir)

    g_vvoo = g_so[nocc:, nocc:, :nocc, :nocc]
    g_vvvv = g_so[nocc:, nocc:, nocc:, nocc:]
    g_oooo = g_so[:nocc, :nocc, :nocc, :nocc]
    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    g_ovoo = g_so[:nocc, nocc:, :nocc, :nocc]
    g_vvov = g_so[nocc:, nocc:, :nocc, nocc:]
    g_ovov = g_so[:nocc, nocc:, :nocc, nocc:]
    g_ooov = g_so[:nocc, :nocc, :nocc, nocc:]
    g_ovvv = g_so[:nocc, nocc:, nocc:, nocc:]

    f_oo = f_so[:nocc, :nocc]
    f_vv = f_so[nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]

    d_oo = -dip[:nocc, :nocc]
    d_vv = -dip[nocc:, nocc:]
    d_ov = -dip[:nocc, nocc:]
    d_vo = -dip[nocc:, :nocc]

    res_t2_20 = np.zeros((nvir, nvir, nocc, nocc))

    res_t2_20 += 1.0 * np.einsum('baji->abij', g_vvoo, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ki,bakj->abij', f_oo, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kj,baki->abij', f_oo, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('bc,acji->abij', f_vv, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ac,bcji->abij', f_vv, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kbji,ak->abij', g_ovoo, t1_10, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kaji,bk->abij', g_ovoo, t1_10, optimize=True)
    res_t2_20 += -1.0 * np.einsum('baic,cj->abij', g_vvov, t1_10, optimize=True)
    res_t2_20 += 1.0 * np.einsum('bajc,ci->abij', g_vvov, t1_10, optimize=True)
    res_t2_20 += -0.5 * np.einsum('klji,balk->abij', g_oooo, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kbic,ackj->abij', g_ovov, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kaic,bckj->abij', g_ovov, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kbjc,acki->abij', g_ovov, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kajc,bcki->abij', g_ovov, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('bacd,dcji->abij', g_vvvv, t2_20, optimize=True)
    #res_t2_20 += 1.0 * np.einsum('I,baji->abijI', G, t2_21, optimize=True)
    res_t2_20 += -1.0 * np.einsum('bi,aj->abij', d_vo, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ai,bj->abij', d_vo, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('bj,ai->abij', d_vo, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('aj,bi->abij', d_vo, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ki,bakj->abij', d_oo, t2_21, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kj,baki->abij', d_oo, t2_21, optimize=True)
    res_t2_20 += -1.0 * np.einsum('bc,acji->abij', d_vv, t2_21, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ac,bcji->abij', d_vv, t2_21, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,ci,bakj->abij', f_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,cj,baki->abij', f_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,bk,acji->abij', f_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,ak,bcji->abij', f_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klji,bk,al->abij', g_oooo, t1_10, t1_10, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kbic,cj,ak->abij', g_ovov, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kaic,cj,bk->abij', g_ovov, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kbjc,ci,ak->abij', g_ovov, t1_10, t1_10, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kajc,ci,bk->abij', g_ovov, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * np.einsum('bacd,di,cj->abij', g_vvvv, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * t1_01 * np.einsum('ki,bakj->abij', d_oo, t2_20, optimize=True)
    res_t2_20 += -1.0 * t1_01 *  np.einsum('kj,baki->abij', d_oo, t2_20, optimize=True)
    res_t2_20 += -1.0 * t1_01 *  np.einsum('bc,acji->abij', d_vv, t2_20, optimize=True)
    res_t2_20 += 1.0 * t1_01 *  np.einsum('ac,bcji->abij', d_vv, t2_20, optimize=True)
    res_t2_20 += 0.5 * np.einsum('klic,cj,balk->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klic,bk,aclj->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klic,ak,bclj->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klic,ck,balj->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('kljc,ci,balk->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kljc,bk,acli->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kljc,ak,bcli->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kljc,ck,bali->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kbcd,di,ackj->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kacd,di,bckj->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kbcd,dj,acki->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kacd,dj,bcki->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('kbcd,ak,dcji->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kbcd,dk,acji->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += 0.5 * np.einsum('kacd,bk,dcji->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kacd,dk,bcji->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('klcd,bdji,aclk->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += 0.5 * np.einsum('klcd,adji,bclk->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += 0.25 * np.einsum('klcd,dcji,balk->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('klcd,baki,dclj->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klcd,bdki,aclj->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klcd,adki,bclj->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('klcd,dcki,balj->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ki,bk,aj->abij', d_oo, t1_10, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('ki,ak,bj->abij', d_oo, t1_10, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kj,bk,ai->abij', d_oo, t1_10, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kj,ak,bi->abij', d_oo, t1_10, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('bc,ci,aj->abij', d_vv, t1_10, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ac,ci,bj->abij', d_vv, t1_10, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('bc,cj,ai->abij', d_vv, t1_10, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('ac,cj,bi->abij', d_vv, t1_10, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_21, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_21, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_21, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_21, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,ck,baji->abij', d_ov, t1_10, t2_21, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,bcki,aj->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,acki,bj->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,bckj,ai->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,ackj,bi->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klic,cj,bk,al->abij', g_ooov, t1_10, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kljc,ci,bk,al->abij', g_ooov, t1_10, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kbcd,di,cj,ak->abij', g_ovvv, t1_10, t1_10, t1_10, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kacd,di,cj,bk->abij', g_ovvv, t1_10, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * t1_01 *  np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * t1_01 *  np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * t1_01 *  np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * t1_01 *  np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('klcd,di,cj,balk->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klcd,di,bk,aclj->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klcd,di,ak,bclj->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klcd,di,ck,balj->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klcd,dj,bk,acli->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klcd,dj,ak,bcli->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klcd,dj,ck,bali->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('klcd,bk,al,dcji->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klcd,bk,dl,acji->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klcd,ak,dl,bcji->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,ci,bk,aj->abij', d_ov, t1_10, t1_10, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,ci,ak,bj->abij', d_ov, t1_10, t1_10, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,cj,bk,ai->abij', d_ov, t1_10, t1_10, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,cj,ak,bi->abij', d_ov, t1_10, t1_10, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klcd,di,cj,bk,al->abij', g_oovv, t1_10, t1_10, t1_10, t1_10, optimize=True)


    t2_20 += np.einsum('abij,iajb -> abij', res_t2_20, e_denom, optimize=True)

    return t2_20

def ccsd_t2_21(f_so, g_so, dip, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nvir = t2_20.shape[0]
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]
    eps_vir_p_w = eps[nocc:] + w
    e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir.reshape(-1, 1, 1) - eps_vir_p_w)

    g_vvvv = g_so[nocc:, nocc:, nocc:, nocc:]
    g_oooo = g_so[:nocc, :nocc, :nocc, :nocc]
    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    g_ovoo = g_so[:nocc, nocc:, :nocc, :nocc]
    g_vvov = g_so[nocc:, nocc:, :nocc, nocc:]
    g_ovov = g_so[:nocc, nocc:, :nocc, nocc:]
    g_ooov = g_so[:nocc, :nocc, :nocc, nocc:]
    g_ovvv = g_so[:nocc, nocc:, nocc:, nocc:]

    f_oo = f_so[:nocc, :nocc]
    f_vv = f_so[nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]

    d_oo = -dip[:nocc, :nocc]
    d_vv = -dip[nocc:, nocc:]
    d_ov = -dip[:nocc, nocc:]
    d_vo = -dip[nocc:, :nocc]

    res_t2_21 = np.zeros((nvir, nvir, nocc, nocc))

    res_t2_21 += 1.0 * np.einsum('ki,bakj->abij', d_oo, t2_20, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kj,baki->abij', d_oo, t2_20, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bc,acji->abij', d_vv, t2_20, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ac,bcji->abij', d_vv, t2_20, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ki,bakj->abij', f_oo, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kj,baki->abij', f_oo, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bc,acji->abij', f_vv, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ac,bcji->abij', f_vv, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbji,ak->abij', g_ovoo, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kaji,bk->abij', g_ovoo, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('baic,cj->abij', g_vvov, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('bajc,ci->abij', g_vvov, t2_11, optimize = True)
    res_t2_21 += 1.0 * w * np.einsum('baji->abij', t2_21, optimize = True)
    #res_t2_21 += 1.0 * G * np.einsum('J,baji->abij', t2_22, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bi,aj->abij', d_vo, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ai,bj->abij', d_vo, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('bj,ai->abij', d_vo, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('aj,bi->abij', d_vo, t2_12, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klji,balk->abij', g_oooo, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbic,ackj->abij', g_ovov, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kaic,bckj->abij', g_ovov, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbjc,acki->abij', g_ovov, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kajc,bcki->abij', g_ovov, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('bacd,dcji->abij', g_vvvv, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ki,bakj->abij', d_oo, t2_22, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kj,baki->abij', d_oo, t2_22, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bc,acji->abij', d_vv, t2_22, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ac,bcji->abij', d_vv, t2_22, optimize = True)
    res_t2_21 += 1.0 * t2_02 * np.einsum('ki,bakj->abij', d_oo, t2_20, optimize = True)
    res_t2_21 += -1.0 * t2_02 * np.einsum('kj,baki->abij', d_oo, t2_20, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += -1.0 * t2_02 * np.einsum('bc,acji->abij', d_vv, t2_20, optimize = True)
    res_t2_21 += 1.0 * t2_02 * np.einsum('ac,bcji->abij', d_vv, t2_20, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ci,bakj->abij', f_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,cj,baki->abij', f_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bk,acji->abij', f_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ak,bcji->abij', f_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,bcji,ak->abij', f_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,acji,bk->abij', f_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,baki,cj->abij', f_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bakj,ci->abij', f_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klji,bk,al->abij', g_oooo, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klji,ak,bl->abij', g_oooo, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbic,cj,ak->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kaic,cj,bk->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbic,ak,cj->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kaic,bk,cj->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbjc,ci,ak->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kajc,ci,bk->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbjc,ak,ci->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kajc,bk,ci->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('bacd,di,cj->abij', g_vvvv, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bacd,dj,ci->abij', g_vvvv, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ki,bk,aj->abij', d_oo, t1_10, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('ki,ak,bj->abij', d_oo, t1_10, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kj,bk,ai->abij', d_oo, t1_10, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kj,ak,bi->abij', d_oo, t1_10, t2_12, optimize = True)
    res_t2_21 += 1.0 * t1_01 * np.einsum('ki,bakj->abij', d_oo, t2_21, optimize = True)
    res_t2_21 += -1.0 * t1_01 * np.einsum('kj,baki->abij', d_oo, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bc,ci,aj->abij', d_vv, t1_10, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ac,ci,bj->abij', d_vv, t1_10, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('bc,cj,ai->abij', d_vv, t1_10, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('ac,cj,bi->abij', d_vv, t1_10, t2_12, optimize = True)
    res_t2_21 += -1.0 * t1_01 * np.einsum('bc,acji->abij', d_vv, t2_21, optimize = True)
    res_t2_21 += 1.0 * t1_01 * np.einsum('ac,bcji->abij', d_vv, t2_21, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klic,cj,balk->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klic,bk,aclj->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klic,ak,bclj->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klic,ck,balj->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('kljc,ci,balk->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kljc,bk,acli->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kljc,ak,bcli->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kljc,ck,bali->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klic,bakj,cl->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klic,bckj,al->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klic,ackj,bl->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klic,balk,cj->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kljc,baki,cl->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kljc,bcki,al->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kljc,acki,bl->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += -0.5 * np.einsum('kljc,balk,ci->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbcd,di,ackj->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kacd,di,bckj->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbcd,dj,acki->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kacd,dj,bcki->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('kbcd,ak,dcji->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbcd,dk,acji->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += 0.5 * np.einsum('kacd,bk,dcji->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kacd,dk,bcji->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbcd,adji,ck->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += -0.5 * np.einsum('kbcd,dcji,ak->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kacd,bdji,ck->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += 0.5 * np.einsum('kacd,dcji,bk->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbcd,adki,cj->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kacd,bdki,cj->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbcd,adkj,ci->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kacd,bdkj,ci->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ck,baji->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bcki,aj->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,acki,bj->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,bckj,ai->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ackj,bi->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += 1.0 * t2_02 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += -1.0 * t2_02 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += 1.0 * t2_02 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += -1.0 * t2_02 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,bdji,aclk->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klcd,adji,bclk->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 0.25 * np.einsum('klcd,dcji,balk->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,baki,dclj->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,bdki,aclj->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,adki,bclj->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,dcki,balj->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klcd,bakj,dcli->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,bdkj,acli->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,adkj,bcli->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klcd,dckj,bali->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 0.25 * np.einsum('klcd,balk,dcji->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,bdlk,acji->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klcd,adlk,bcji->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('ki,bj,ak->abij', d_oo, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ki,aj,bk->abij', d_oo, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kj,bi,ak->abij', d_oo, t2_11, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kj,ai,bk->abij', d_oo, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('bc,ai,cj->abij', d_vv, t2_11, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bc,ci,aj->abij', d_vv, t2_11, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('ac,bi,cj->abij', d_vv, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ac,ci,bj->abij', d_vv, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bi,ackj->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ai,bckj->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += 2.0 * np.einsum('kc,ci,bakj->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,bj,acki->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,aj,bcki->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += -2.0 * np.einsum('kc,cj,baki->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += 2.0 * np.einsum('kc,bk,acji->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += -2.0 * np.einsum('kc,ak,bcji->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ck,baji->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klic,cj,bk,al->abij', g_ooov, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klic,cj,ak,bl->abij', g_ooov, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klic,bk,al,cj->abij', g_ooov, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kljc,ci,bk,al->abij', g_ooov, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kljc,ci,ak,bl->abij', g_ooov, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kljc,bk,al,ci->abij', g_ooov, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbcd,di,cj,ak->abij', g_ovvv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kacd,di,cj,bk->abij', g_ovvv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbcd,di,ak,cj->abij', g_ovvv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kacd,di,bk,cj->abij', g_ovvv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbcd,dj,ak,ci->abij', g_ovvv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kacd,dj,bk,ci->abij', g_ovvv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ci,bk,aj->abij', d_ov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ci,ak,bj->abij', d_ov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,cj,bk,ai->abij', d_ov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,cj,ak,bi->abij', d_ov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_21 += 1.0 * t1_01 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * t1_01 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * t1_01 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * t1_01 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * t1_01 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * t1_01 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * t1_01 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * t1_01 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,di,cj,balk->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,di,bk,aclj->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,di,ak,bclj->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,di,ck,balj->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,dj,bk,acli->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,dj,ak,bcli->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,dj,ck,bali->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,bk,al,dcji->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,bk,dl,acji->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,ak,dl,bcji->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,di,bakj,cl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,di,bckj,al->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,di,ackj,bl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,di,balk,cj->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,dj,baki,cl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,dj,bcki,al->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,dj,acki,bl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,bk,adji,cl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,bk,dcji,al->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,ak,bdji,cl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,dk,bcji,al->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klcd,ak,dcji,bl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,dk,acji,bl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,bk,adli,cj->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,ak,bdli,cj->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,dk,bali,cj->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klcd,dj,balk,ci->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,bk,adlj,ci->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,ak,bdlj,ci->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,dk,balj,ci->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ci,bj,ak->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ci,aj,bk->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,cj,bi,ak->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,cj,ai,bk->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,bk,ai,cj->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bk,ci,aj->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ak,bi,cj->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ak,ci,bj->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)

    t2_21 += np.einsum('abij,iajb -> abij', res_t2_21, e_denom, optimize=True)

    return t2_21

def ccsd_t2_22(f_so, g_so, dip, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nvir = t2_20.shape[0]
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]
    eps_vir_p_2w = eps[nocc:] + 2 * w
    e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir.reshape(-1, 1, 1) - eps_vir_p_2w)

    g_vvvv = g_so[nocc:, nocc:, nocc:, nocc:]
    g_oooo = g_so[:nocc, :nocc, :nocc, :nocc]
    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    g_ovoo = g_so[:nocc, nocc:, :nocc, :nocc]
    g_vvov = g_so[nocc:, nocc:, :nocc, nocc:]
    g_ovov = g_so[:nocc, nocc:, :nocc, nocc:]
    g_ooov = g_so[:nocc, :nocc, :nocc, nocc:]
    g_ovvv = g_so[:nocc, nocc:, nocc:, nocc:]

    f_oo = f_so[:nocc, :nocc]
    f_vv = f_so[nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]

    d_oo = -dip[:nocc, :nocc]
    d_vv = -dip[nocc:, nocc:]
    d_ov = -dip[:nocc, nocc:]

    res_t2_22 = np.zeros((nvir, nvir, nocc, nocc))

    res_t2_22 += 1.0 * np.einsum('ki,bakj->abij', f_oo, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kj,baki->abij', f_oo, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bc,acji->abij', f_vv, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ac,bcji->abij', f_vv, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbji,ak->abij', g_ovoo, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kaji,bk->abij', g_ovoo, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('baic,cj->abij', g_vvov, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('bajc,ci->abij', g_vvov, t2_12, optimize = True)
    res_t2_22 += 1.0 * w * np.einsum('baji->abij', t2_22, optimize = True)
    res_t2_22 += 1.0 * w * np.einsum('baji->abij', t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ki,bakj->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kj,baki->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ki,bakj->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kj,baki->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bc,acji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ac,bcji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bc,acji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ac,bcji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klji,balk->abij', g_oooo, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbic,ackj->abij', g_ovov, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kaic,bckj->abij', g_ovov, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbjc,acki->abij', g_ovov, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kajc,bcki->abij', g_ovov, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('bacd,dcji->abij', g_vvvv, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bakj->abij', f_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,baki->abij', f_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,acji->abij', f_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bcji->abij', f_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bcji,ak->abij', f_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,acji,bk->abij', f_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,baki,cj->abij', f_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bakj,ci->abij', f_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klji,bk,al->abij', g_oooo, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klji,ak,bl->abij', g_oooo, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbic,cj,ak->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kaic,cj,bk->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbic,ak,cj->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kaic,bk,cj->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbjc,ci,ak->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kajc,ci,bk->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbjc,ak,ci->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kajc,bk,ci->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('bacd,di,cj->abij', g_vvvv, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bacd,dj,ci->abij', g_vvvv, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * t1_01 * np.einsum('ki,bakj->abij', d_oo, t2_22, optimize = True)
    res_t2_22 += -1.0 * t1_01 * np.einsum('kj,baki->abij', d_oo, t2_22, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('ki,bakj->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kj,baki->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('ki,bakj->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kj,baki->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * t1_01 * np.einsum('bc,acji->abij', d_vv, t2_22, optimize = True)
    res_t2_22 += 1.0 * t1_01 * np.einsum('ac,bcji->abij', d_vv, t2_22, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('bc,acji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('ac,bcji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('bc,acji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('ac,bcji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klic,cj,balk->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klic,bk,aclj->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klic,ak,bclj->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klic,ck,balj->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('kljc,ci,balk->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kljc,bk,acli->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kljc,ak,bcli->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kljc,ck,bali->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klic,bakj,cl->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klic,bckj,al->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klic,ackj,bl->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klic,balk,cj->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kljc,baki,cl->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kljc,bcki,al->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kljc,acki,bl->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += -0.5 * np.einsum('kljc,balk,ci->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbcd,di,ackj->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kacd,di,bckj->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbcd,dj,acki->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kacd,dj,bcki->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('kbcd,ak,dcji->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbcd,dk,acji->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += 0.5 * np.einsum('kacd,bk,dcji->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kacd,dk,bcji->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbcd,adji,ck->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += -0.5 * np.einsum('kbcd,dcji,ak->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kacd,bdji,ck->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += 0.5 * np.einsum('kacd,dcji,bk->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbcd,adki,cj->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kacd,bdki,cj->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbcd,adkj,ci->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kacd,bdkj,ci->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,bdji,aclk->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,adji,bclk->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 0.25 * np.einsum('klcd,dcji,balk->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,baki,dclj->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,bdki,aclj->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,adki,bclj->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,dcki,balj->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,bakj,dcli->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,bdkj,acli->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,adkj,bcli->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,dckj,bali->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 0.25 * np.einsum('klcd,balk,dcji->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,bdlk,acji->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,adlk,bcji->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kc,ci,bakj->abij', f_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kc,cj,baki->abij', f_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kc,bk,acji->abij', f_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kc,ak,bcji->abij', f_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klji,bk,al->abij', g_oooo, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kbic,cj,ak->abij', g_ovov, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kaic,cj,bk->abij', g_ovov, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kbjc,ci,ak->abij', g_ovov, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kajc,ci,bk->abij', g_ovov, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('bacd,di,cj->abij', g_vvvv, t2_11, t2_11, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ki,bk,aj->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('ki,ak,bj->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kj,bk,ai->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kj,ak,bi->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ki,bk,aj->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('ki,ak,bj->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kj,bk,ai->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kj,ak,bi->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('ki,bj,ak->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ki,aj,bk->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kj,bi,ak->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kj,ai,bk->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bc,ci,aj->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ac,ci,bj->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('bc,cj,ai->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('ac,cj,bi->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bc,ci,aj->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ac,ci,bj->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('bc,cj,ai->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('ac,cj,bi->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('bc,ai,cj->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('ac,bi,cj->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bc,aj,ci->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ac,bj,ci->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klic,cj,balk->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klic,bk,aclj->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klic,ak,bclj->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klic,ck,balj->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kljc,ci,balk->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kljc,bk,acli->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kljc,ak,bcli->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kljc,ck,bali->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kbcd,di,ackj->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kacd,di,bckj->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kbcd,dj,acki->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kacd,dj,bcki->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbcd,ak,dcji->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kbcd,dk,acji->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kacd,bk,dcji->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kacd,dk,bcji->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ck,baji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ck,baji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bi,ackj->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ai,bckj->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bj,acki->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,aj,bcki->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bcki,aj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,acki,bj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bckj,ai->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ackj,bi->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bcki,aj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,acki,bj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bckj,ai->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ackj,bi->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,baji,ck->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klic,cj,bk,al->abij', g_ooov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klic,cj,ak,bl->abij', g_ooov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klic,bk,al,cj->abij', g_ooov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kljc,ci,bk,al->abij', g_ooov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kljc,ci,ak,bl->abij', g_ooov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kljc,bk,al,ci->abij', g_ooov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbcd,di,cj,ak->abij', g_ovvv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kacd,di,cj,bk->abij', g_ovvv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbcd,di,ak,cj->abij', g_ovvv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kacd,di,bk,cj->abij', g_ovvv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbcd,dj,ak,ci->abij', g_ovvv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kacd,dj,bk,ci->abij', g_ovvv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * t1_01 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * t1_01 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * t1_01 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * t1_01 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * t1_01 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * t1_01 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * t1_01 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * t1_01 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,bdji,aclk->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,adji,bclk->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,dcji,balk->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,baki,dclj->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,bdki,aclj->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,adki,bclj->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dcki,balj->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,di,cj,balk->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,di,bk,aclj->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,di,ak,bclj->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,di,ck,balj->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dj,bk,acli->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,dj,ak,bcli->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dj,ck,bali->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,bk,al,dcji->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,bk,dl,acji->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,ak,dl,bcji->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,di,bakj,cl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,di,bckj,al->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,di,ackj,bl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,di,balk,cj->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,dj,baki,cl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dj,bcki,al->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,dj,acki,bl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,bk,adji,cl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,bk,dcji,al->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,ak,bdji,cl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,dk,bcji,al->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,ak,dcji,bl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dk,acji,bl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,bk,adli,cj->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,ak,bdli,cj->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,dk,bali,cj->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,dj,balk,ci->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,bk,adlj,ci->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,ak,bdlj,ci->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dk,balj,ci->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klic,cj,bk,al->abij', g_ooov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klic,bk,cj,al->abij', g_ooov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klic,ak,cj,bl->abij', g_ooov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kljc,ci,bk,al->abij', g_ooov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kljc,bk,ci,al->abij', g_ooov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kljc,ak,ci,bl->abij', g_ooov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kbcd,di,cj,ak->abij', g_ovvv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kacd,di,cj,bk->abij', g_ovvv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kbcd,dj,ci,ak->abij', g_ovvv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kacd,dj,ci,bk->abij', g_ovvv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kbcd,ak,di,cj->abij', g_ovvv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kacd,bk,di,cj->abij', g_ovvv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bk,aj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ci,ak,bj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,ci,aj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,ci,bj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,bk,ai->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,cj,ak,bi->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bk,cj,ai->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ak,cj,bi->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bk,aj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ci,ak,bj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,ci,aj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,ci,bj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,bk,ai->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,cj,ak,bi->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bk,cj,ai->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ak,cj,bi->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ci,bj,ak->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,aj,bk->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,cj,bi,ak->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,ai,bk->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bk,ai,cj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ak,bi,cj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,aj,ci->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bj,ci->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 2.0 * t1_01 * np.einsum('kc,ci,bakj->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * t1_01 * np.einsum('kc,cj,baki->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * t1_01 * np.einsum('kc,bk,acji->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * t1_01 * np.einsum('kc,ak,bcji->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,di,cj,balk->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,di,bk,aclj->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,di,ak,bclj->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,di,ck,balj->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,dj,ci,balk->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,bk,di,aclj->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,ak,di,bclj->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,dk,ci,balj->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,dj,bk,acli->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,dj,ak,bcli->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,dj,ck,bali->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,bk,dj,acli->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,ak,dj,bcli->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,dk,cj,bali->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,bk,al,dcji->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,bk,dl,acji->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,ak,bl,dcji->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,dk,bl,acji->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,ak,dl,bcji->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,dk,al,bcji->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,bdji,ak,cl->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,adji,bk,cl->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dcji,bk,al->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,baki,dj,cl->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,bdki,cj,al->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,adki,cj,bl->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,bakj,di,cl->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,bdkj,ci,al->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,adkj,ci,bl->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,balk,di,cj->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kc,bi,cj,ak->abij', d_ov, t2_11, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kc,ci,bj,ak->abij', d_ov, t2_11, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kc,ai,cj,bk->abij', d_ov, t2_11, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kc,ci,aj,bk->abij', d_ov, t2_11, t2_11, t2_11, optimize = True)

    t2_22 += np.einsum('abij,iajb -> abij', res_t2_22, e_denom, optimize=True)

    return t2_22

def ccsd_t1_10(f_so, g_so, dip, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nvir = t2_20.shape[0]
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]
    e_denom = 1 / (eps_occ.reshape(-1, 1) - eps_vir)

    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    g_ovov = g_so[:nocc, nocc:, :nocc, nocc:]
    g_ooov = g_so[:nocc, :nocc, :nocc, nocc:]
    g_ovvv = g_so[:nocc, nocc:, nocc:, nocc:]

    f_oo = f_so[:nocc, :nocc]
    f_vv = f_so[nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]
    f_vo = f_so[nocc:, :nocc]

    d_oo = -dip[:nocc, :nocc]
    d_vv = -dip[nocc:, nocc:]
    d_ov = -dip[:nocc, nocc:]
    d_vo = -dip[nocc:, :nocc]

    res_t1_10 = np.zeros((nvir, nocc))

    res_t1_10 += 1.0 * np.einsum('ai->ai', f_vo, optimize=True)
    res_t1_10 += -1.0 * np.einsum('ji,aj->ai', f_oo, t1_10, optimize=True)
    res_t1_10 += 1.0 * np.einsum('ab,bi->ai', f_vv, t1_10, optimize=True)
    res_t1_10 += 1.0 * t1_01 * np.einsum('ai->ai', d_vo, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jb,abji->ai', f_ov, t2_20, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jaib,bj->ai', g_ovov, t1_10, optimize=True)
    res_t1_10 += 0.5 * np.einsum('jkib,abkj->ai', g_ooov, t2_20, optimize=True)
    res_t1_10 += -0.5 * np.einsum('jabc,cbji->ai', g_ovvv, t2_20, optimize=True)
    #res_t1_10 += 1.0 * np.einsum('ai->aiI', G, t2_11, optimize=True)
    res_t1_10 += -1.0 * np.einsum('ji,aj->ai', d_oo, t2_11, optimize=True)
    res_t1_10 += 1.0 * np.einsum('ab,bi->ai', d_vv, t2_11, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jb,bi,aj->ai', f_ov, t1_10, t1_10, optimize=True)
    res_t1_10 += -1.0 * t1_01 * np.einsum('ji,aj->ai', d_oo, t1_10, optimize=True)
    res_t1_10 += 1.0 * t1_01 * np.einsum('ab,bi->ai', d_vv, t1_10, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jb,abji->ai', d_ov, t2_21, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jkib,aj,bk->ai', g_ooov, t1_10, t1_10, optimize=True)
    res_t1_10 += 1.0 * np.einsum('jabc,ci,bj->ai', g_ovvv, t1_10, t1_10, optimize=True)
    res_t1_10 += -1.0 * t1_01 * np.einsum('jb,abji->ai', d_ov, t2_20, optimize=True)
    res_t1_10 += -0.5 * np.einsum('jkbc,ci,abkj->ai', g_oovv, t1_10, t2_20, optimize=True)
    res_t1_10 += -0.5 * np.einsum('jkbc,aj,cbki->ai', g_oovv, t1_10, t2_20, optimize=True)
    res_t1_10 += 1.0 * np.einsum('jkbc,cj,abki->ai', g_oovv, t1_10, t2_20, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_11, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_11, optimize=True)
    res_t1_10 += 1.0 * np.einsum('jb,bj,ai->ai', d_ov, t1_10, t2_11, optimize=True)
    res_t1_10 += -1.0 * t1_01 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t1_10, optimize=True)
    res_t1_10 += 1.0 * np.einsum('jkbc,ci,aj,bk->ai', g_oovv, t1_10, t1_10, t1_10, optimize=True)

    t1_10 += np.einsum('ai,ia -> ai', res_t1_10, e_denom, optimize=True)

    return t1_10

def ccsd_t1_01(f_so, g_so, dip, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]

    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]
    d_ov = -dip[:nocc, nocc:]

    res_t1_01 = 0
    G = 0
    #res_t1_01 += 1.0 * G
    res_t1_01 += 1.0 * w * t1_01
    res_t1_01 += 1.0 * G * t2_02
    res_t1_01 += 1.0 * np.einsum('ia,ai->', d_ov, t1_10, optimize=True)
    res_t1_01 += 1.0 * np.einsum('ia,ai->', f_ov, t2_11, optimize=True)
    res_t1_01 += 1.0 * np.einsum('ia,ai->', d_ov, t2_12, optimize=True)
    res_t1_01 += 1.0 * t2_02 * np.einsum('ia,ai->', d_ov, t1_10, optimize=True)
    res_t1_01 += 0.25 * np.einsum('ijab,baji->', g_oovv, t2_21, optimize=True)
    res_t1_01 += 1.0 * t1_01 * np.einsum('ia,ai->', d_ov, t2_11, optimize=True)
    res_t1_01 += -1.0 * np.einsum('ijab,bi,aj->', g_oovv, t1_10, t2_11, optimize=True)

    if w == 0:
        t1_01 = 0
    else:
        t1_01 += -res_t1_01 / w

    return t1_01

def ccsd_t2_02(f_so, g_so, dip, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]

    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]
    d_ov = -dip[:nocc, nocc:]

    res_t2_02 = 0

    res_t2_02 += 1.0 * w * t2_02
    res_t2_02 += 1.0 * w * t2_02
    res_t2_02 += 1.0 * np.einsum('ia,ai->', f_ov, t2_12)
    res_t2_02 += 1.0 * np.einsum('ia,ai->', d_ov, t2_11)
    res_t2_02 += 1.0 * np.einsum('ia,ai->', d_ov, t2_11)
    res_t2_02 += 0.25 * np.einsum('ijab,baji->', g_oovv, t2_22)
    res_t2_02 += 1.0 * t1_01 * np.einsum('ia,ai->', d_ov, t2_12)
    res_t2_02 += 1.0 * t2_02 * np.einsum('ia,ai->', d_ov, t2_11)
    res_t2_02 += 1.0 * t2_02 * np.einsum('ia,ai->', d_ov, t2_11)
    res_t2_02 += -1.0 * np.einsum('ijab,bi,aj->', g_oovv, t1_10, t2_12)
    res_t2_02 += -1.0 * np.einsum('ijab,bi,aj->', g_oovv, t2_11, t2_11)

    if w == 0:
        t2_02 = 0
    else:
        t2_02 += -res_t2_02 / (2 * w)

    return t2_02

def ccsd_t2_11(f_so, g_so, dip, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nvir = t2_20.shape[0]
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:] + w
    e_denom = 1 / (eps_occ.reshape(-1, 1) - eps_vir)

    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    g_ovov = g_so[:nocc, nocc:, :nocc, nocc:]
    g_ooov = g_so[:nocc, :nocc, :nocc, nocc:]
    g_ovvv = g_so[:nocc, nocc:, nocc:, nocc:]

    f_oo = f_so[:nocc, :nocc]
    f_vv = f_so[nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]

    d_oo = -dip[:nocc, :nocc]
    d_vv = -dip[nocc:, nocc:]
    d_ov = -dip[:nocc, nocc:]
    d_vo = -dip[nocc:, :nocc]

    res_t2_11 = np.zeros((nvir, nocc))

    res_t2_11 += 1.0 * np.einsum('ai->ai', d_vo, optimize = True)
    res_t2_11 += -1.0 * np.einsum('ji,aj->ai', d_oo, t1_10, optimize = True)
    res_t2_11 += 1.0 * t2_02 * np.einsum('ai->ai', d_vo, optimize = True)
    res_t2_11 += 1.0 * np.einsum('ab,bi->ai', d_vv, t1_10, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,abji->ai', d_ov, t2_20, optimize = True)
    res_t2_11 += -1.0 * np.einsum('ji,aj->ai', f_oo, t2_11, optimize = True)
    res_t2_11 += 1.0 * np.einsum('ab,bi->ai', f_vv, t2_11, optimize = True)
    res_t2_11 += 1.0 * w * np.einsum('ai->ai', t2_11, optimize = True)
    #res_t2_11 += 1.0 * G * np.einsum('ai->ai', t2_12, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,abji->ai', f_ov, t2_21, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jaib,bj->ai', g_ovov, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('ji,aj->ai', d_oo, t2_12, optimize = True)
    res_t2_11 += 1.0 * np.einsum('ab,bi->ai', d_vv, t2_12, optimize = True)
    res_t2_11 += -1.0 * t2_02 * np.einsum('ji,aj->ai', d_oo, t1_10, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t1_10, optimize = True)
    res_t2_11 += 1.0 * t2_02 * np.einsum('ab,bi->ai', d_vv, t1_10, optimize = True)
    res_t2_11 += 0.5 * np.einsum('jkib,abkj->ai', g_ooov, t2_21, optimize = True)
    res_t2_11 += -0.5 * np.einsum('jabc,cbji->ai', g_ovvv, t2_21, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,abji->ai', d_ov, t2_22, optimize = True)
    res_t2_11 += -1.0 * t2_02 * np.einsum('jb,abji->ai', d_ov, t2_20, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,bi,aj->ai', f_ov, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,aj,bi->ai', f_ov, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * t1_01 * np.einsum('ji,aj->ai', d_oo, t2_11, optimize = True)
    res_t2_11 += 1.0 * t1_01 * np.einsum('ab,bi->ai', d_vv, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jkib,aj,bk->ai', g_ooov, t1_10, t2_11, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jkib,bj,ak->ai', g_ooov, t1_10, t2_11, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jabc,ci,bj->ai', g_ovvv, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jabc,cj,bi->ai', g_ovvv, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_12, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_12, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jb,bj,ai->ai', d_ov, t1_10, t2_12, optimize = True)
    res_t2_11 += -1.0 * t1_01 * np.einsum('jb,abji->ai', d_ov, t2_21, optimize = True)
    res_t2_11 += -1.0 * t2_02 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t1_10, optimize = True)
    res_t2_11 += -0.5 * np.einsum('jkbc,ci,abkj->ai', g_oovv, t1_10, t2_21, optimize = True)
    res_t2_11 += -0.5 * np.einsum('jkbc,aj,cbki->ai', g_oovv, t1_10, t2_21, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jkbc,cj,abki->ai', g_oovv, t1_10, t2_21, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jkbc,acji,bk->ai', g_oovv, t2_20, t2_11, optimize = True)
    res_t2_11 += 0.5 * np.einsum('jkbc,cbji,ak->ai', g_oovv, t2_20, t2_11, optimize = True)
    res_t2_11 += 0.5 * np.einsum('jkbc,ackj,bi->ai', g_oovv, t2_20, t2_11, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jb,ai,bj->ai', d_ov, t2_11, t2_11, optimize = True)
    res_t2_11 += -2.0 * np.einsum('jb,bi,aj->ai', d_ov, t2_11, t2_11, optimize = True)
    res_t2_11 += -1.0 * t1_01 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * t1_01 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jkbc,ci,aj,bk->ai', g_oovv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jkbc,ci,bj,ak->ai', g_oovv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jkbc,aj,ck,bi->ai', g_oovv, t1_10, t1_10, t2_11, optimize = True)

    t2_11 += np.einsum('ai,ia -> ai', res_t2_11, e_denom, optimize=True)

    return t2_11

def ccsd_t2_12(f_so, g_so, dip, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nvir = t2_20.shape[0]
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:] + 2 * w
    e_denom = 1 / (eps_occ.reshape(-1, 1) - eps_vir)

    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    g_ovov = g_so[:nocc, nocc:, :nocc, nocc:]
    g_ooov = g_so[:nocc, :nocc, :nocc, nocc:]
    g_ovvv = g_so[:nocc, nocc:, nocc:, nocc:]

    f_oo = f_so[:nocc, :nocc]
    f_vv = f_so[nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]

    d_oo = -dip[:nocc, :nocc]
    d_vv = -dip[nocc:, nocc:]
    d_ov = -dip[:nocc, nocc:]

    res_t2_12 = np.zeros((nvir, nocc))

    res_t2_12 += -1.0 * np.einsum('ji,aj->ai', f_oo, t2_12, optimize = True)
    res_t2_12 += 1.0 * np.einsum('ab,bi->ai', f_vv, t2_12, optimize = True)
    res_t2_12 += 1.0 * w * np.einsum('ai->ai', t2_12, optimize = True)
    res_t2_12 += 1.0 * w * np.einsum('ai->ai', t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('ji,aj->ai', d_oo, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('ji,aj->ai', d_oo, t2_11, optimize = True)
    res_t2_12 += 1.0 * np.einsum('ab,bi->ai', d_vv, t2_11, optimize = True)
    res_t2_12 += 1.0 * np.einsum('ab,bi->ai', d_vv, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,abji->ai', f_ov, t2_22, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jaib,bj->ai', g_ovov, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,abji->ai', d_ov, t2_21, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,abji->ai', d_ov, t2_21, optimize = True)
    res_t2_12 += 0.5 * np.einsum('jkib,abkj->ai', g_ooov, t2_22, optimize = True)
    res_t2_12 += -0.5 * np.einsum('jabc,cbji->ai', g_ovvv, t2_22, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,bi,aj->ai', f_ov, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,aj,bi->ai', f_ov, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * t1_01 * np.einsum('ji,aj->ai', d_oo, t2_12, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('ji,aj->ai', d_oo, t2_11, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('ji,aj->ai', d_oo, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += 1.0 * t1_01 * np.einsum('ab,bi->ai', d_vv, t2_12, optimize = True)
    res_t2_12 += 1.0 * t2_02 * np.einsum('ab,bi->ai', d_vv, t2_11, optimize = True)
    res_t2_12 += 1.0 * t2_02 * np.einsum('ab,bi->ai', d_vv, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jkib,aj,bk->ai', g_ooov, t1_10, t2_12, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jkib,bj,ak->ai', g_ooov, t1_10, t2_12, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jabc,ci,bj->ai', g_ovvv, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jabc,cj,bi->ai', g_ovvv, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * t1_01 * np.einsum('jb,abji->ai', d_ov, t2_22, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('jb,abji->ai', d_ov, t2_21, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('jb,abji->ai', d_ov, t2_21, optimize = True)
    res_t2_12 += -0.5 * np.einsum('jkbc,ci,abkj->ai', g_oovv, t1_10, t2_22, optimize = True)
    res_t2_12 += -0.5 * np.einsum('jkbc,aj,cbki->ai', g_oovv, t1_10, t2_22, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jkbc,cj,abki->ai', g_oovv, t1_10, t2_22, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jkbc,acji,bk->ai', g_oovv, t2_20, t2_12, optimize = True)
    res_t2_12 += 0.5 * np.einsum('jkbc,cbji,ak->ai', g_oovv, t2_20, t2_12, optimize = True)
    res_t2_12 += 0.5 * np.einsum('jkbc,ackj,bi->ai', g_oovv, t2_20, t2_12, optimize = True)
    res_t2_12 += -2.0 * np.einsum('jb,bi,aj->ai', f_ov, t2_11, t2_11, optimize = True)
    res_t2_12 += -2.0 * np.einsum('jkib,aj,bk->ai', g_ooov, t2_11, t2_11, optimize = True)
    res_t2_12 += 2.0 * np.einsum('jabc,ci,bj->ai', g_ovvv, t2_11, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jb,bj,ai->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jb,bj,ai->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jb,ai,bj->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += -1.0 * t1_01 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * t1_01 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jkbc,ci,abkj->ai', g_oovv, t2_11, t2_21, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jkbc,aj,cbki->ai', g_oovv, t2_11, t2_21, optimize = True)
    res_t2_12 += 2.0 * np.einsum('jkbc,cj,abki->ai', g_oovv, t2_11, t2_21, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jkbc,ci,aj,bk->ai', g_oovv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jkbc,ci,bj,ak->ai', g_oovv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jkbc,aj,ck,bi->ai', g_oovv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_12 += -2.0 * t1_01 * np.einsum('jb,bi,aj->ai', d_ov, t2_11, t2_11, optimize = True)
    res_t2_12 += 2.0 * np.einsum('jkbc,ci,aj,bk->ai', g_oovv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_12 += 2.0 * np.einsum('jkbc,aj,ci,bk->ai', g_oovv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_12 += 2.0 * np.einsum('jkbc,cj,bi,ak->ai', g_oovv, t1_10, t2_11, t2_11, optimize = True)

    t2_12 += np.einsum('ai,ia -> ai', res_t2_12, e_denom, optimize=True)

    return t2_12

def spin_block_tei(I):
    '''
    Spin blocks 2-electron integrals
    Using np.kron, we project I and I tranpose into the space of the 2x2 ide
    The result is our 2-electron integral tensor in spin orbital notation
    '''
    identity = np.eye(2)
    I = np.kron(identity, I)
    return np.kron(identity, I.T)

def ao_to_mo_transform_full(h_core, g_ao, C):
    tmp = np.einsum('pi,pqrs->iqrs', C, g_ao, optimize=True)
    tmp = np.einsum('iqrs,rj->iqjs', tmp, C, optimize=True)
    tmp = np.einsum('qa,iqjs->iajs', C, tmp, optimize=True)
    g_mo = np.einsum('iajs,sb->iajb', tmp, C, optimize=True)

    h_mo = np.einsum('pi,pq,qj -> ij', C, h_core, C, optimize=True)

    return h_mo, g_mo

def ao_to_mo_transform_full2(h_core, g_ao, X, Y):
    tmp = np.einsum('pi,pqrs->iqrs', X, g_ao, optimize=True)
    tmp = np.einsum('iqrs,rj->iqjs', tmp, Y, optimize=True)
    tmp = np.einsum('qa,iqjs->iajs', X, tmp, optimize=True)
    g_mo = np.einsum('iajs,sb->iajb', tmp, Y, optimize=True)

    h_mo = np.einsum('pi,pq,qj -> ij', X, h_core, Y, optimize=True)

    return h_mo, g_mo

def get_fock(h, g, nocc):
    fock = np.zeros((h.shape[0], h.shape[1]))
    fock += h
    fock += np.einsum('piqi -> pq', g[:, :nocc, :, :nocc], optimize=True)

    return fock







###################################################################




def setup_calculation( LEVEL, mol, lambda_cav, omega ):

    lambda_x = lambda_cav[0]
    lambda_y = lambda_cav[1]
    lambda_z = lambda_cav[2]

    #Up to singles photonic excitations
    do_t1_01 = False
    do_t2_11 = False
    do_t2_21 = False
    #Up to double photonic excitations
    do_t2_02 = False
    do_t2_12 = False
    do_t2_22 = False

    if ( LEVEL == "CCSD-S1-U21" ): # CCSD-S1-U21
        do_t1_01 = True
        do_t2_11 = True
        do_t2_21 = True
    elif ( LEVEL == "CCSD-S2-U12" ): # CCSD-S2-U12
        do_t1_01 = True
        do_t2_11 = True
        do_t2_02 = True
        do_t2_12 = True
    elif ( LEVEL == "CCSD-S2-U22" ): # CCSD-S2-U22
        do_t1_01 = True
        do_t2_11 = True
        do_t2_21 = True 
        do_t2_02 = True
        do_t2_12 = True
        do_t2_22 = True

    # Why can't we do CCSD-S1-U11 ? ~BMW


    #conventional
    if do_t1_01 == False and do_t2_11 == False and do_t2_21 == False and do_t2_02 == False and do_t2_12 == False and do_t2_22 == False:
        print("doing conventional CCSD")
    #Deprince, White, Full doubles
    if do_t1_01 == True and do_t2_11 == True and do_t2_21 == True and do_t2_02 == False and do_t2_12 == False and do_t2_22 == False:
        print("doing QED-CCSD-1 (or QED-CCSD-21)")
    if do_t1_01 == True and do_t2_11 == True and do_t2_21 == False and do_t2_02 == True and do_t2_12 == True and do_t2_22 == False:
        print("doing QED-CCSD-White (or QED-CCSD-12)")
    if do_t1_01 == True and do_t2_11 == True and do_t2_21 == True and do_t2_02 == True and do_t2_12 == True and do_t2_22 == True:
        print("doing QED-CCSD-Full (or QED-CCSD-22)")


    # ==> Compute static 1e- and 2e- quantities with Psi4 <==
    # Class instantiation
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
    obs = wfn.basisset()
    mints = psi4.core.MintsHelper(obs)
    basis = wfn.basisset()
    SCF_E_psi, scf_wfn = psi4.energy('scf', return_wfn=True)

    # Overlap matrix
    S = np.asarray(mints.ao_overlap())

    # Number of basis Functions & doubly occupied orbitals
    nbf = S.shape[0]
    nocc = wfn.nalpha()

    print('Number of occupied orbitals: %3d' % (nocc))
    print('Number of basis functions: %3d' % (nbf))

    zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

    # Initialize the JK object
    jk = psi4.core.JK.build(wfn.basisset())
    jk.set_memory(int(1.25e8))  # 1GB
    jk.initialize()
    jk.print_header()

    # Build core Hamiltonian
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    H_core = T + V

    # Orthogonalization matrix
    X = mints.ao_overlap()
    X.power(-0.5, 1.e-16)
    X = np.asarray(X)

    # nuclear-neculear repulsion energy
    E_nuc = mol.nuclear_repulsion_energy()

    #guess denisty matrix
    C = np.asarray(scf_wfn.Ca())
    Cocc = C[:, :nocc]
    D = Cocc.dot(Cocc.transpose())

    diff_SCF_E = 1
    E_old = 0
    E_threshold = 1e-10
    MAXITER = 500

    #
    # photonic initialization
    #

    # dipole integrals
    mu_x_ao = np.asarray(mints.so_dipole()[0])
    mu_y_ao = np.asarray(mints.so_dipole()[1])
    mu_z_ao = np.asarray(mints.so_dipole()[2])

    # get quadrupole integrals
    quadrupole_xx = 0.5 * lambda_x * lambda_x * np.asarray(mints.so_quadrupole()[0])
    quadrupole_xy = 1.0 * lambda_x * lambda_y * np.asarray(mints.so_quadrupole()[1])
    quadrupole_xz = 1.0 * lambda_x * lambda_z * np.asarray(mints.so_quadrupole()[2])
    quadrupole_yy = 0.5 * lambda_y * lambda_y * np.asarray(mints.so_quadrupole()[3])
    quadrupole_yz = 1.0 * lambda_y * lambda_z * np.asarray(mints.so_quadrupole()[4])
    quadrupole_zz = 0.5 * lambda_z * lambda_z * np.asarray(mints.so_quadrupole()[5])

    quadrupole_x_lambda2_tot = (quadrupole_xx + quadrupole_xy + quadrupole_xz +
                                quadrupole_yy + quadrupole_yz + quadrupole_zz)

    dipole_x_lambda_tot = (lambda_x * mu_x_ao +
                        lambda_y * mu_y_ao +
                        lambda_z * mu_z_ao)

    I = np.asarray(mints.ao_eri())
    # Begin Iterations
    for scf_iter in range(1, MAXITER + 1):

        # Efficient Fock build
        jk.C_left_add(psi4.core.Matrix.from_array(Cocc))
        jk.C_right_add(psi4.core.Matrix.from_array(Cocc))
        jk.compute()
        jk.C_clear()
        J = np.array(jk.J()[0])
        K = np.array(jk.K()[0])

        F = H_core + 2 * J - K

        oei = np.zeros((H_core.shape[0], H_core.shape[1]))

        #electron + nuclear dipole moment
        mu_x_mo = np.einsum('pq, pq ->', 2 * D, mu_x_ao, optimize=True)
        mu_y_mo = np.einsum('pq, pq ->', 2 * D, mu_y_ao, optimize=True)
        mu_z_mo = np.einsum('pq, pq ->', 2 * D, mu_z_ao, optimize=True)

        mu_x_lambda_tot_mo = -(lambda_x * mu_x_mo + lambda_y * mu_y_mo + lambda_z * mu_z_mo)

        DSE = 0.5 * mu_x_lambda_tot_mo * mu_x_lambda_tot_mo

        oei = dipole_x_lambda_tot * mu_x_lambda_tot_mo
        oei -= quadrupole_x_lambda2_tot
        F += oei

        scaled_mu = np.einsum('pq, pq -> ', D, dipole_x_lambda_tot, optimize=True)
        F += 2 * scaled_mu * dipole_x_lambda_tot
        F -= np.einsum('pr, qs, rs -> pq', dipole_x_lambda_tot, dipole_x_lambda_tot, D, optimize=True)

        E_new = (np.einsum('pq,pq->', (oei + H_core + F), D, optimize=True) + E_nuc + DSE)


        print('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E' % (scf_iter, E_new, E_new - E_old))

        # SCF Converged?
        if (abs(E_new - E_old) < E_threshold):
            break
        E_old = E_new

        e, Ct = np.linalg.eigh(X.dot(F).dot(X))
        C = X.dot(Ct)
        Cocc = C[:, :nocc]
        D = Cocc.dot(Cocc.transpose())

        if (scf_iter == MAXITER):
            psi4.core.clean()
            raise Exception("Maximum number of SCF iterations exceeded.")

    # Post iterations
    print('\nSCF converged.')

    print('E_QED_HF          = %4.15f' % (E_new))
    print('HF  (psi4)        = %4.15f' % (SCF_E_psi))

    ###Starting with the QED Coupled Cluster routine
    scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
    # Get basis and orbital information
    nalpha = scf_wfn.nalpha()  # Number of alpha electrons
    nbeta = scf_wfn.nbeta()  # Number of beta electrons
    nao = C.shape[0]  # Number of atomic orbitals
    nmo = C.shape[1]  # Number of molecular orbitals
    nocc = nalpha + nbeta  # Number of occupied spin orbitals
    nso = 2 * nmo  # Total number of spin orbitals
    nvir = nso - nocc  # Number of virtual spin orbitals

    # getting eri integrals

    I = np.asarray(mints.ao_eri())

    g_ao = I + (lambda_x * lambda_x * np.einsum('pq,rs->pqrs', mu_x_ao, mu_x_ao)
            + lambda_y * lambda_y * np.einsum('pq,rs->pqrs', mu_y_ao, mu_y_ao)
            + lambda_z * lambda_z * np.einsum('pq,rs->pqrs', mu_z_ao, mu_z_ao))

    # ==> core Hamiltoniam <==
    h = H_core + oei

    h_mo_sf, g_mo_sf = ao_to_mo_transform_full(h, g_ao, C)

    @jit(nopython=True)
    def sf_to_so(g_mo_sf):
        g_mo = np.zeros((nso, nso, nso, nso))
        for p in range(nso):
            for q in range(nso):
                for r in range(nso):
                    for s in range(nso):
                        value1 = g_mo_sf[p// 2, q// 2, r// 2, s// 2] * (p % 2 == q % 2) * (r % 2 == s % 2)
                        value2 = g_mo_sf[p// 2, s// 2, r// 2, q// 2] * (p % 2 == s % 2) * (q % 2 == r % 2)
                        g_mo[p,q,r,s] = value1 - value2
        return g_mo
    g_mo = sf_to_so(g_mo_sf)

    g_mo = g_mo.transpose(0, 2, 1, 3)

    #defined as g_{pq}^{X} in White's paper (https://aip.scitation.org/doi/pdf/10.1063/5.0033132)
    coupling_factor_x = lambda_x*math.sqrt(omega/2)
    coupling_factor_y = lambda_y*math.sqrt(omega/2)
    coupling_factor_z = lambda_z*math.sqrt(omega/2)

    dipole_x_tr = np.einsum('pi,pq,qj -> ij', C, mu_x_ao, C, optimize=True)
    dipole_y_tr = np.einsum('pi,pq,qj -> ij', C, mu_y_ao, C, optimize=True)
    dipole_z_tr = np.einsum('pi,pq,qj -> ij', C, mu_z_ao, C, optimize=True)

    dip_sf = (dipole_x_tr * coupling_factor_x
            + dipole_y_tr * coupling_factor_y
            + dipole_z_tr * coupling_factor_z)

    G = 0
    f_tmp = F.dot(C)
    f = C.transpose().dot(f_tmp)

    f_mo = np.zeros((nso,nso))
    dip = np.zeros((nso,nso))
    for p in range(nso):
        for q in range(nso):
            f_mo[p, q] = f[p // 2, q//2] * (p % 2 == q % 2)
            dip[p, q] = dip_sf[p // 2, q // 2] * (p % 2 == q % 2)

    #pure singles amplitudes electron/photon
    t1_10 = np.zeros((nvir, nocc))
    t1_01 = 0
    #pure double amplitudes electron/photon
    t2_20 = np.zeros((nvir, nvir, nocc, nocc))
    t2_02 = 0
    #mixed double amplitudes electron-photon
    t2_11 = np.zeros((nvir, nocc))
    t2_21 = np.zeros((nvir, nvir, nocc, nocc))
    t2_12 = np.zeros((nvir, nocc))
    t2_22 = np.zeros((nvir, nvir, nocc, nocc))

    E_CCSD_old = 0
    tol = 1e-8
    MAXITER = 50
    time_average = 0

    ### Setup DIIS
    diis_vals_t1_10 = [t1_10.copy()]
    diis_vals_t2_20 = [t2_20.copy()]
    #up to singles photonic excitations
    if do_t2_11:
        diis_vals_t2_11 = [t2_11.copy()]
    if do_t2_21:
        diis_vals_t2_21 = [t2_21.copy()]
    #up to double photonic excitations
    if do_t2_12:
        diis_vals_t2_12 = [t2_12.copy()]
    if do_t2_22:
        diis_vals_t2_22 = [t2_22.copy()]

    diis_errors = []
    max_diis = 20

    print('Starting with the CCSD calculation:\n')
    print('Iter  Energy(CCSD)      E_diff    time(sec)')
    for ccsd_iter in range(1, MAXITER + 1):

        t_start = time.time()

        # Save new amplitudes
        oldt1_10 = t1_10.copy()
        oldt2_20 = t2_20.copy()
        if do_t2_11:
            oldt2_11 = t2_11.copy()
        if do_t2_21:
            oldt2_21 = t2_21.copy()
        if do_t2_12:
            oldt2_12 = t2_12.copy()
        if do_t2_22:
            oldt2_22 = t2_22.copy()

        #singles
        t1_10 = ccsd_t1_10(f_mo, g_mo, dip, G, omega, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22)
        if do_t1_01:
            t1_01 = ccsd_t1_01(f_mo, g_mo, dip, G, omega, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22)
        #pure doubles
        t2_20 = ccsd_t2_20(f_mo, g_mo, dip, G, omega, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22)
        if do_t2_02:
            t2_02 = ccsd_t2_02(f_mo, g_mo, dip, G, omega, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22)
        #mixed doubles
        if do_t2_11:
            t2_11 = ccsd_t2_11(f_mo, g_mo, dip, G, omega, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22)
        if do_t2_21:
            t2_21 = ccsd_t2_21(f_mo, g_mo, dip, G, omega, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22)
        if do_t2_12:
            t2_12 = ccsd_t2_12(f_mo, g_mo, dip, G, omega, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22)
        if do_t2_22:
            t2_22 = ccsd_t2_22(f_mo, g_mo, dip, G, omega, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22)

        #E_CCSD_new += 1.0 * G * t1_01
        E_CCSD_new = 1.0 * np.einsum('ia,ai->', f_mo[:nocc, nocc:], t1_10)
        E_CCSD_new = 0.25 * np.einsum('ijab,baji->', g_mo[:nocc, :nocc, nocc:, nocc:], t2_20)
        E_CCSD_new -= 1.0 * np.einsum('ia,ai->', dip[:nocc, nocc:], t2_11)
        E_CCSD_new -= 1.0 * t1_01 * np.einsum('ia,ai->', dip[:nocc, nocc:], t1_10)
        E_CCSD_new += -0.5 * np.einsum('ijab,bi,aj->', g_mo[:nocc, :nocc, nocc:, nocc:], t1_10, t1_10)

        t_total = time.time() - t_start
        time_average += t_total
        print('%3d:  %4.10f  %1.5E   %.3f' % (ccsd_iter, E_CCSD_new, \
                                            abs(E_CCSD_new - E_CCSD_old), t_total))

        if (abs(E_CCSD_new - E_CCSD_old) < tol):
            break

        # Add DIIS vectors
        diis_vals_t1_10.append(t1_10.copy())
        diis_vals_t2_20.append(t2_20.copy())
        if do_t2_11:
            diis_vals_t2_11.append(t2_11.copy())
        if do_t2_21:
            diis_vals_t2_21.append(t2_21.copy())
        if do_t2_12:
            diis_vals_t2_12.append(t2_12.copy())
        if do_t2_22:
            diis_vals_t2_22.append(t2_22.copy())

        # Build new error vector
        error_t1_10 = (t1_10 - oldt1_10).ravel()
        error_t2_20 = (t2_20 - oldt2_20).ravel()
        if do_t2_11:
            error_t2_11 = (t2_11 - oldt2_11).ravel()
        if do_t2_21:
            error_t2_21 = (t2_21 - oldt2_21).ravel()
        if do_t2_12:
            error_t2_12 = (t2_12 - oldt2_12).ravel()
        if do_t2_22:
            error_t2_22 = (t2_22 - oldt2_22).ravel()

        #conventional
        if do_t1_01 == False and do_t2_11 == False and do_t2_21 == False and do_t2_02 == False and do_t2_12 == False and do_t2_22 == False:
            diis_errors.append(np.concatenate((error_t1_10, error_t2_20)))
        #Deprince, White, Full doubles
        if do_t1_01 == True and do_t2_11 == True and do_t2_21 == True and do_t2_02 == False and do_t2_12 == False and do_t2_22 == False:
            diis_errors.append(np.concatenate((error_t1_10, error_t2_20, error_t2_11, error_t2_21)))
        if do_t1_01 == True and do_t2_11 == True and do_t2_21 == False and do_t2_02 == True and do_t2_12 == True and do_t2_22 == False:
            diis_errors.append(np.concatenate((error_t1_10, error_t2_20, error_t2_11, error_t2_12)))
        if do_t1_01 == True and do_t2_11 == True and do_t2_21 == True and do_t2_02 == True and do_t2_12 == True and do_t2_22 == True:
            diis_errors.append(np.concatenate((error_t1_10, error_t2_20, error_t2_11, error_t2_21, error_t2_12, error_t2_22)))


        E_CCSD_old = E_CCSD_new


        if ccsd_iter >= 1:

            # Limit size of DIIS vector
            if (len(diis_vals_t1_10) > max_diis):
                del diis_vals_t1_10[0]
                del diis_vals_t2_20[0]
                if do_t2_11:
                    del diis_vals_t2_11[0]
                if do_t2_21:
                    del diis_vals_t2_21[0]
                if do_t2_12:
                    del diis_vals_t2_12[0]
                if do_t2_22:
                    del diis_vals_t2_22[0]
                del diis_errors[0]

            diis_size = len(diis_vals_t1_10) - 1

            # Build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
            B = np.ones((diis_size + 1, diis_size + 1)) * -1
            B[-1, -1] = 0

            for n1, e1 in enumerate(diis_errors):
                for n2, e2 in enumerate(diis_errors):
                    # Vectordot the error vectors
                    B[n1, n2] = np.dot(e1, e2)

            B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

            # Build residual vector, [Pulay:1980:393], Eqn. 6, RHS
            resid = np.zeros(diis_size + 1)
            resid[-1] = -1

            # Solve Pulay equations, [Pulay:1980:393], Eqn. 6
            ci = np.linalg.solve(B, resid)

            # Calculate new amplitudes
            t1_10[:] = 0
            t2_20[:] = 0
            if do_t2_11:
                t2_11[:] = 0
            if do_t2_21:
                t2_21[:] = 0
            if do_t2_12:
                t2_12[:] = 0
            if do_t2_22:
                t2_22[:] = 0
            for num in range(diis_size):
                t1_10 += ci[num] * diis_vals_t1_10[num + 1]
                t2_20 += ci[num] * diis_vals_t2_20[num + 1]
                if do_t2_11:
                    t2_11 += ci[num] * diis_vals_t2_11[num + 1]
                if do_t2_21:
                    t2_21 += ci[num] * diis_vals_t2_21[num + 1]
                if do_t2_12:
                    t2_12 += ci[num] * diis_vals_t2_12[num + 1]
                if do_t2_22:
                    t2_22 += ci[num] * diis_vals_t2_22[num + 1]
        # End DIIS amplitude update


    ccsd_e, ccsd_wfn = psi4.energy('ccsd', return_wfn=True)

    print('E_QED_CCSD  (Corr.)  = %4.15f' % (E_CCSD_new))
    print('E_QED_CCSD tot       = %4.15f' % (E_CCSD_new+E_new))
    print('E_CCSD_psi4 (Corr.)  = %4.15f' % (ccsd_e - SCF_E_psi))
    print('E_CCSD_psi4 tot      = %4.15f' % (ccsd_e))

    sp.call(f"echo '{float(sys.argv[5])} {E_CCSD_new+E_new}' >> GS_ENERGY.dat", shell=True)

    #reference QED-CCSD-21/STO-3G for a given molecule E(QED-CCSD) = -262.416986187232396



















def main():

    # Choose level of CC
    if ( len(sys.argv) == 6 ):
        LEVEL  = sys.argv[1] # "CCSD-S0-U20", "CCSD-S1-U21", "CCSD-S2-U12", "CCSD-S2-U22"
        if ( LEVEL == "CCSD" ): LEVEL = "CCSD-S0-U20"
        omega  = float(sys.argv[2])/27.2114 # a.u., This is the variable the code uses
        LAMBDA = float(sys.argv[3]) # a.u.
        E_POL  = np.array([ float(j) for j in sys.argv[4] ]) # 100, 010, 100, or any other combination
        E_POL  = E_POL / np.linalg.norm(E_POL)
        lambda_cav = E_POL * LAMBDA # This is the variable the code uses
    else:
        print("\n\tWARNING !!!")
        print("\tNeed to provide level of theory (LOT), cavity frequency (eV), coupling strength (a.u.), and polarization:")
        print("\tLOT Options:\n\t\tCCSD  CCSD-S1-U21  CCSD-S2-U12  CCSD-S2-U22")
        print("\tPolarization Options:\n\t\t100 010 001 or any combonation of integers")
        print("\tDefault: Do standard CCSD calculation without photon.")
        LEVEL = "CCSD"
        omega = 0.1
        lambda_cav = np.array([0,0,0])

    print(f"Settings:\n\tLOT = {LEVEL}\n\twc = {round(omega*27.2114,6)} eV\n\tlambda = {np.round(lambda_cav,6)} a.u.\n\tEquivalent A0 = LAMBDA / sqrt(2 wc) = {np.round(lambda_cav/np.sqrt(2*omega),2)})")

    # ==> Set Basic Psi4 Options <==
    # Memory specification
    psi4.set_memory( "60 GB" )
    psi4.core.set_num_threads(1)

    build_superfunctional = psi4.driver.dft.build_superfunctional # Is this required ??? ~BMW

    # Set output file
    psi4.core.set_output_file('psi4.out', False)

    # The units of position are Angstroms ~ BMW
    Rx = float(sys.argv[5])
    Rx = (Rx - Rx/2) * 0.529
    mol = psi4.geometry("""
    0 1

    H  -%2.8f 0.00000000 0.00000000
    H  %2.8f 0.00000000 0.00000000
    
    symmetry c1
    no_reorient
    nocom
    """ % (Rx,Rx))

    # Set computation options
    psi4.set_options({'basis': 'cc-pVQZ',
                    'scf_type': 'pk',
                    'e_convergence': 1e-8})


    # Do all the preliminary integrals and perform scQED-CCSD-Sn-U2n
    setup_calculation( LEVEL, mol, lambda_cav, omega )





if ( __name__ == "__main__" ):
    main()
