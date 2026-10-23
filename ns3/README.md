# ns-3 validation

`wsn-validate.cc` replays a schedule exported by the Python simulator (cluster heads, cluster membership and the nodes that report in each round) over the IEEE 802.15.4 model of ns-3 (O-QPSK 2.4 GHz PHY, unslotted CSMA/CA, acknowledgements, up to 3 retransmissions, SINR-based error model). Malicious cluster heads drop received packets with the same per-node probability as in the Python model. The path loss is a three-segment log-distance model with exponent 2 up to 87.7 m and 4 beyond, mirroring the first-order radio model, and the transmit power of every
frame is set to reach the receiver 10 dB above -85 dBm. The program prints one CSV line:
`rounds,generated,rx_at_ch,dropped_mal,delivered,data_tx_attempts,mac_fail,delay_ms`.

## Build (tested with ns-3.44, GCC 13, Linux)

```bash
# from the repository root
wget https://www.nsnam.org/releases/ns-allinone-3.44.tar.bz2
tar xjf ns-allinone-3.44.tar.bz2
mv ns-allinone-3.44/ns-3.44 ns-3
cd ns-3
./ns3 configure --build-profile=optimized --disable-examples --disable-tests \
    --enable-modules="lr-wpan;energy;mobility;flow-monitor;propagation;spectrum;network;core"
mkdir -p scratch/wsn-validate
cp ../ns3/wsn-validate.cc scratch/wsn-validate/
./ns3 build wsn-validate
cd ..
```

This produces `ns-3/build/scratch/wsn-validate/ns3.44-wsn-validate-optimized`.

## Run the validation

```bash
python run_validation.py          # 4 methods x 5 seeds x first 300 rounds
```

`run_validation.py` looks for ns-3 in `./ns-3` (set `NS3_DIR` to use another location), runs the Python simulator with 100-byte data packets (the IEEE 802.15.4 frame limit is 127 bytes), exports each schedule and replays it in three ns-3 settings: one channel per cluster, one channel per cluster with Nakagami fading (m = 3), and one channel shared by all clusters. Results are written
to `results/validation/validation_runs.csv` and `validation_summary.csv`.

Program options: `--trace=<file>`, `--mode=channels|shared|serial`, `--fadingM=<m>` (0 = no
fading), `--slot=<s>`, `--margin=<dB>`, `--targetRx=<dBm>`, `--ackPower=<dBm>`, `--seed=<n>`.
