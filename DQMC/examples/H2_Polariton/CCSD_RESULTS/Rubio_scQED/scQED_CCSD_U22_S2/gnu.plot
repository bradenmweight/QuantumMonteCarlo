
set xrange[0.4:6]
#set xrange[0.4:3]
set yrange[-1.2:-0.75]

p "A0_0.0/GS_ENERGY.dat" u 1:2 title "QED-CCSD/cc-pVQZ (Rubio) (A0 = 0.0 a.u.)" @wl
rep "A0_0.1/GS_ENERGY.dat" u 1:2 title "QED-CCSD/cc-pVQZ (Rubio) (A0 = 0.1 a.u.)" @wl
rep "A0_0.2/GS_ENERGY.dat" u 1:2 title "QED-CCSD/cc-pVQZ (Rubio) (A0 = 0.2 a.u.)" @wl
rep "A0_0.3/GS_ENERGY.dat" u 1:2 title "QED-CCSD/cc-pVQZ (Rubio) (A0 = 0.3 a.u.)" @wl
rep "A0_0.4/GS_ENERGY.dat" u 1:2 title "QED-CCSD/cc-pVQZ (Rubio) (A0 = 0.4 a.u.)" @wl
rep "A0_0.5/GS_ENERGY.dat" u 1:2 title "QED-CCSD/cc-pVQZ (Rubio) (A0 = 0.5 a.u.)" @wl

rep "../../../A0_SCAN/EPOL_x/DATA_dt_0.01_0.01_NW_10_6_NSTEPS_2500_5000/PES_Production_WC_20.0.dat" u 1:($2-20/27.2114/2+0.003) title "DQMC (Weight) (A0 = 0.0 a.u.)" @wp
rep "../../../A0_SCAN/EPOL_x/DATA_dt_0.01_0.01_NW_10_6_NSTEPS_2500_5000/PES_Production_WC_20.0.dat" u 1:($3-20/27.2114/2+0.003) title "DQMC (Weight) (A0 = 0.1 a.u.)" @wp
rep "../../../A0_SCAN/EPOL_x/DATA_dt_0.01_0.01_NW_10_6_NSTEPS_2500_5000/PES_Production_WC_20.0.dat" u 1:($4-20/27.2114/2+0.003) title "DQMC (Weight) (A0 = 0.2 a.u.)" @wp
rep "../../../A0_SCAN/EPOL_x/DATA_dt_0.01_0.01_NW_10_6_NSTEPS_2500_5000/PES_Production_WC_20.0.dat" u 1:($5-20/27.2114/2+0.003) title "DQMC (Weight) (A0 = 0.3 a.u.)" @wp
rep "../../../A0_SCAN/EPOL_x/DATA_dt_0.01_0.01_NW_10_6_NSTEPS_2500_5000/PES_Production_WC_20.0.dat" u 1:($6-20/27.2114/2+0.003) title "DQMC (Weight) (A0 = 0.4 a.u.)" @wp
rep "../../../A0_SCAN/EPOL_x/DATA_dt_0.01_0.01_NW_10_6_NSTEPS_2500_5000/PES_Production_WC_20.0.dat" u 1:($7-20/27.2114/2+0.003) title "DQMC (Weight) (A0 = 0.5 a.u.)" @wp


rep "../../../A0_SCAN/EPOL_x/DATA_classical_DSE/PES_Production_WC_20.0.dat" u 1:($2-20/27.2114/2+0.003) title  "" @wl
rep "../../../A0_SCAN/EPOL_x/DATA_classical_DSE/PES_Production_WC_20.0.dat" u 1:($3-20/27.2114/2+0.003) title  "" @wl
rep "../../../A0_SCAN/EPOL_x/DATA_classical_DSE/PES_Production_WC_20.0.dat" u 1:($4-20/27.2114/2+0.003) title  "" @wl
rep "../../../A0_SCAN/EPOL_x/DATA_classical_DSE/PES_Production_WC_20.0.dat" u 1:($5-20/27.2114/2+0.003) title  "" @wl
rep "../../../A0_SCAN/EPOL_x/DATA_classical_DSE/PES_Production_WC_20.0.dat" u 1:($6-20/27.2114/2+0.003) title  "" @wl
rep "../../../A0_SCAN/EPOL_x/DATA_classical_DSE/PES_Production_WC_20.0.dat" u 1:($7-20/27.2114/2+0.003) title  "" @wl
rep "../../../A0_SCAN/EPOL_x/DATA_classical_DSE/PES_Production_WC_20.0.dat" u 1:($8-20/27.2114/2+0.003) title  "" @wl
rep "../../../A0_SCAN/EPOL_x/DATA_classical_DSE/PES_Production_WC_20.0.dat" u 1:($9-20/27.2114/2+0.003) title  "" @wl
rep "../../../A0_SCAN/EPOL_x/DATA_classical_DSE/PES_Production_WC_20.0.dat" u 1:($10-20/27.2114/2+0.003) title "" @wl
rep "../../../A0_SCAN/EPOL_x/DATA_classical_DSE/PES_Production_WC_20.0.dat" u 1:($11-20/27.2114/2+0.003) title "" @wl
rep "../../../A0_SCAN/EPOL_x/DATA_classical_DSE/PES_Production_WC_20.0.dat" u 1:($12-20/27.2114/2+0.003) title "" @wl

