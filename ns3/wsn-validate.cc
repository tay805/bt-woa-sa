/*
 * Trace-driven packet-level validation of the wsnsim round-based simulator.
 *
 * Replays a schedule exported by wsnsim/ns3trace.py (cluster heads, membership and
 * reporting nodes per round) over the full IEEE 802.15.4 stack of ns-3 (O-QPSK 2.4 GHz PHY,
 * unslotted CSMA/CA, ACKs, macMaxFrameRetries = 3, SINR-based error model, inter-cluster
 * interference). Malicious cluster heads drop received member packets with their
 * per-node probability (grey-hole), exactly as in the Python model.
 *
 * Radio/propagation: three-segment log-distance loss mirroring the first-order radio model
 * (exponent 2 up to d0 = 87.7 m, exponent 4 beyond), reference loss 40.05 dB at 1 m (2.4 GHz),
 * optional Nakagami-m fading. Transmit power is set per frame to reach the receiver at
 * targetRx + margin (power control, the assumption behind the first-order amplifier term).
 *
 * Timing per round: members of every cluster transmit in TDMA slots, then each CH forwards
 * one aggregate frame to the BS in its own slot, then direct-to-BS nodes transmit.
 * Inter-cluster separation (--mode):
 *   channels (default): each cluster uses its own 802.15.4 channel (11..25), BS on 26 --
 *                       the FDMA counterpart of LEACH's per-cluster CDMA codes;
 *   shared:  all clusters transmit in parallel on one channel (worst-case interference);
 *   serial:  one channel, clusters take turns (no inter-cluster overlap, longer rounds).
 *
 * Output (stdout, one CSV line):
 *   rounds,generated,rx_at_ch,dropped_mal,delivered,data_tx_attempts,mac_fail,delay_ms
 */
#include "ns3/constant-position-mobility-model.h"
#include "ns3/core-module.h"
#include "ns3/lr-wpan-module.h"
#include "ns3/network-module.h"
#include "ns3/propagation-delay-model.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/single-model-spectrum-channel.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;
using namespace ns3::lrwpan;

struct NodeInfo
{
    double x, y;
    bool mal;
    double drop;
    int phase;
};

struct RoundInfo
{
    int r;
    std::vector<int> chs;
    std::map<int, std::vector<int>> members; // ch -> members (id order); -1 = direct to BS
    std::set<int> senders;
};

struct PktInfo
{
    int round;
    int src;
    int count;      // readings carried (aggregate frames)
    double genTime; // round start (readings generated at round start)
};

static int N = 0;
static double bsX = 0, bsY = 0;
static int dataBytes = 100;
static int attackPeriod = 0;
static double attackOn = 0.5;
static std::vector<NodeInfo> info;
static std::vector<RoundInfo> rounds;
static std::vector<Ptr<LrWpanNetDevice>> devs; // index N = BS
static Ptr<PropagationLossModel> lossModel;
static double targetRxDbm = -85.0;
static double marginDb = 10.0;
static Ptr<UniformRandomVariable> rng;
static std::string mode = "channels";
static std::vector<int> nodeChan;
static const int BS_CHANNEL = 26;
static double ackPowerDbm = 20.0;

static void
SetChan(int idx, int ch)
{
    if (nodeChan[idx] == ch)
        return;
    Ptr<PhyPibAttributes> a = Create<PhyPibAttributes>();
    a->phyCurrentChannel = ch;
    devs[idx]->GetPhy()->PlmeSetAttributeRequest(PhyPibAttributeIdentifier::phyCurrentChannel, a);
    nodeChan[idx] = ch;
    // ns-3 switches the transceiver off on a channel change: turn the receiver back on, and
    // give ACK frames (sent with the current PSD) enough power on the new channel
    LrWpanSpectrumValueHelper helper;
    devs[idx]->GetPhy()->SetTxPowerSpectralDensity(helper.CreateTxPowerSpectralDensity(ackPowerDbm, ch));
    devs[idx]->GetPhy()->PlmeSetTRXStateRequest(IEEE_802_15_4_PHY_RX_ON);
}

static std::map<uint64_t, PktInfo> pktInfo;
static std::map<int, std::vector<uint64_t>> chBuf; // ch -> kept member packet uids (current round)
static std::map<int, std::set<uint64_t>> chSeen;  // ch -> uids already received (dedupe retries)
static std::set<uint64_t> bsSeen;
static std::map<int, bool> chOwn;

static uint64_t generated = 0, rxAtCh = 0, droppedMal = 0, delivered = 0, dataTx = 0, macFail = 0;
static double delaySum = 0.0;

static Mac16Address
Addr(int idx)
{
    char buf[8];
    std::snprintf(buf, sizeof(buf), "%02x:%02x", ((idx + 1) >> 8) & 0xff, (idx + 1) & 0xff);
    return Mac16Address(buf);
}

static bool
Attacking(int ch, int r)
{
    if (!info[ch].mal)
        return false;
    if (attackPeriod <= 0)
        return true;
    return ((r + info[ch].phase) % attackPeriod) < attackOn * attackPeriod;
}

static void
Send(int src, int dst, Ptr<Packet> p)
{
    Ptr<MobilityModel> a = devs[src]->GetPhy()->GetMobility();
    Ptr<MobilityModel> b = devs[dst]->GetPhy()->GetMobility();
    double pl = -lossModel->CalcRxPower(0.0, a, b); // path loss in dB for 0 dBm
    double txDbm = targetRxDbm + marginDb + pl;
    LrWpanSpectrumValueHelper helper;
    devs[src]->GetPhy()->SetTxPowerSpectralDensity(helper.CreateTxPowerSpectralDensity(txDbm, nodeChan[src]));
    McpsDataRequestParams params;
    params.m_dstPanId = 0;
    params.m_srcAddrMode = SHORT_ADDR;
    params.m_dstAddrMode = SHORT_ADDR;
    params.m_dstAddr = Addr(dst);
    params.m_msduHandle = 0;
    params.m_txOptions = TX_OPTION_ACK;
    devs[src]->GetMac()->McpsDataRequest(params, p);
}

static void
MemberSend(int m, int ch, int r, double t0)
{
    Ptr<Packet> p = Create<Packet>(dataBytes);
    pktInfo[p->GetUid()] = PktInfo{r, m, 1, t0};
    Send(m, ch, p);
}

static void
ChFlush(int ch, int r, double t0)
{
    int n = (int)chBuf[ch].size() + (chOwn[ch] ? 1 : 0);
    chBuf[ch].clear();
    chSeen[ch].clear();
    if (n == 0)
        return;
    if (mode == "channels")
        SetChan(ch, BS_CHANNEL);
    Ptr<Packet> p = Create<Packet>(dataBytes);
    pktInfo[p->GetUid()] = PktInfo{r, ch, n, t0};
    Send(ch, N, p);
}

static void
DirectSend(int i, int r, double t0)
{
    Ptr<Packet> p = Create<Packet>(dataBytes);
    pktInfo[p->GetUid()] = PktInfo{r, i, 1, t0};
    if (mode == "channels")
        SetChan(i, BS_CHANNEL);
    Send(i, N, p);
}

static void
DataIndication(int idx, McpsDataIndicationParams params, Ptr<Packet> p)
{
    auto it = pktInfo.find(p->GetUid());
    if (it == pktInfo.end())
        return;
    const PktInfo& pi = it->second;
    if (idx == N)
    { // base station
        if (!bsSeen.insert(p->GetUid()).second)
            return;
        delivered += pi.count;
        delaySum += pi.count * (Simulator::Now().GetSeconds() - pi.genTime);
        return;
    }
    // a cluster head receiving a member reading
    if (!chSeen[idx].insert(p->GetUid()).second)
        return;
    rxAtCh++;
    if (Attacking(idx, pi.round) && rng->GetValue() < info[idx].drop)
    {
        droppedMal++;
        return;
    }
    chBuf[idx].push_back(p->GetUid());
}

static void
DataConfirm(McpsDataConfirmParams params)
{
    if (params.m_status != MacStatus::SUCCESS)
        macFail++;
}

static void
PhyTxBegin(Ptr<const Packet> p)
{
    if (p->GetSize() > 20) // data frames only (ACK frames are 5 bytes)
        dataTx++;
}

static bool
LoadTrace(const std::string& path)
{
    std::ifstream f(path);
    if (!f)
        return false;
    std::string line;
    RoundInfo* cur = nullptr;
    while (std::getline(f, line))
    {
        std::istringstream s(line);
        std::string tag;
        s >> tag;
        if (tag == "HEADER")
        {
            std::string k;
            int R;
            s >> k >> N >> k >> bsX >> bsY >> k >> dataBytes >> k >> R >> k >> attackPeriod >> k >>
                attackOn;
            info.resize(N);
        }
        else if (tag == "NODE")
        {
            int id, mal;
            NodeInfo ni;
            s >> id >> ni.x >> ni.y >> mal >> ni.drop >> ni.phase;
            ni.mal = mal != 0;
            info[id] = ni;
        }
        else if (tag == "ROUND")
        {
            rounds.emplace_back();
            cur = &rounds.back();
            s >> cur->r;
        }
        else if (tag == "CH")
        {
            int h;
            while (s >> h)
                cur->chs.push_back(h);
        }
        else if (tag == "M")
        {
            int m, h;
            while (s >> m >> h)
                cur->members[h].push_back(m);
        }
        else if (tag == "S")
        {
            int i;
            while (s >> i)
                cur->senders.insert(i);
        }
    }
    return true;
}

int
main(int argc, char* argv[])
{
    std::string trace;
    double slot = 0.010;
    double fadingM = 0.0;
    uint32_t seed = 1;
    CommandLine cmd(__FILE__);
    cmd.AddValue("trace", "schedule exported by wsnsim/ns3trace.py", trace);
    cmd.AddValue("slot", "TDMA slot length (s)", slot);
    cmd.AddValue("margin", "power-control margin above target RX power (dB)", marginDb);
    cmd.AddValue("targetRx", "target RX power (dBm)", targetRxDbm);
    cmd.AddValue("fadingM", "Nakagami-m fading parameter (0 = no fading)", fadingM);
    cmd.AddValue("seed", "RNG run number", seed);
    cmd.AddValue("ackPower", "transmit power used for ACK frames (dBm)", ackPowerDbm);
    cmd.AddValue("mode", "inter-cluster separation: channels | shared | serial", mode);
    cmd.Parse(argc, argv);
    RngSeedManager::SetSeed(12345);
    RngSeedManager::SetRun(seed);

    if (!LoadTrace(trace))
    {
        std::cerr << "cannot read trace " << trace << std::endl;
        return 1;
    }
    rng = CreateObject<UniformRandomVariable>();

    Ptr<SingleModelSpectrumChannel> channel = CreateObject<SingleModelSpectrumChannel>();
    Ptr<ThreeLogDistancePropagationLossModel> logd = CreateObject<ThreeLogDistancePropagationLossModel>();
    logd->SetAttribute("Distance0", DoubleValue(1.0));
    logd->SetAttribute("Distance1", DoubleValue(87.7));
    logd->SetAttribute("Distance2", DoubleValue(100000.0));
    logd->SetAttribute("Exponent0", DoubleValue(2.0));
    logd->SetAttribute("Exponent1", DoubleValue(4.0));
    logd->SetAttribute("Exponent2", DoubleValue(4.0));
    logd->SetAttribute("ReferenceLoss", DoubleValue(40.05));
    lossModel = logd;
    channel->AddPropagationLossModel(logd);
    if (fadingM > 0)
    {
        Ptr<NakagamiPropagationLossModel> nak = CreateObject<NakagamiPropagationLossModel>();
        nak->SetAttribute("m0", DoubleValue(fadingM));
        nak->SetAttribute("m1", DoubleValue(fadingM));
        nak->SetAttribute("m2", DoubleValue(fadingM));
        channel->AddPropagationLossModel(nak); // power control uses mean path loss only
    }
    channel->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());

    for (int i = 0; i <= N; i++)
    {
        Ptr<Node> node = CreateObject<Node>();
        Ptr<LrWpanNetDevice> dev = CreateObject<LrWpanNetDevice>();
        dev->SetChannel(channel);
        node->AddDevice(dev);
        dev->SetAddress(Addr(i));
        dev->GetMac()->SetPanId(0);
        dev->GetMac()->SetMacMaxFrameRetries(3);
        Ptr<ConstantPositionMobilityModel> mob = CreateObject<ConstantPositionMobilityModel>();
        mob->SetPosition(i < N ? Vector(info[i].x, info[i].y, 0) : Vector(bsX, bsY, 0));
        dev->GetPhy()->SetMobility(mob);
        dev->GetMac()->SetMcpsDataIndicationCallback(MakeBoundCallback(&DataIndication, i));
        dev->GetMac()->SetMcpsDataConfirmCallback(MakeCallback(&DataConfirm));
        dev->GetPhy()->TraceConnectWithoutContext("PhyTxBegin", MakeCallback(&PhyTxBegin));
        devs.push_back(dev);
    }
    nodeChan.assign(N + 1, -1);
    for (int i = 0; i <= N; i++)
        SetChan(i, (mode == "channels" && i == N) ? BS_CHANNEL : 11);

    double t = 0.1;
    for (const RoundInfo& R : rounds)
    {
        size_t maxMembers = 0;
        for (int h : R.chs)
            if (R.members.count(h))
                maxMembers = std::max(maxMembers, R.members.at(h).size());
        for (size_t c = 0; c < R.chs.size(); c++)
        {
            int h = R.chs[c];
            int chan = mode == "channels" ? 11 + (int)(c % 15) : 11;
            double off = mode == "serial" ? c * (maxMembers + 1) * slot : 0.0;
            bool own = R.senders.count(h) > 0;
            if (own)
                generated++;
            std::vector<int> grp{h};
            if (R.members.count(h))
                grp.insert(grp.end(), R.members.at(h).begin(), R.members.at(h).end());
            Simulator::Schedule(Seconds(t - slot / 2), [grp, chan, h, own]() {
                for (int i : grp)
                    SetChan(i, chan);
                chOwn[h] = own;
            });
            if (!R.members.count(h))
                continue;
            const std::vector<int>& mem = R.members.at(h);
            for (size_t k = 0; k < mem.size(); k++)
            {
                int m = mem[k];
                if (!R.senders.count(m))
                    continue;
                generated++;
                Simulator::Schedule(Seconds(t + off + k * slot), &MemberSend, m, h, R.r, t);
            }
        }
        if (mode == "serial")
            maxMembers = R.chs.size() * (maxMembers + 1);
        double tch = t + (maxMembers + 1) * slot;
        for (size_t j = 0; j < R.chs.size(); j++)
            Simulator::Schedule(Seconds(tch + j * slot), &ChFlush, R.chs[j], R.r, t);
        double tdir = tch + (R.chs.size() + 1) * slot;
        size_t d = 0;
        if (R.members.count(-1))
            for (int i : R.members.at(-1))
                if (R.senders.count(i))
                {
                    generated++;
                    Simulator::Schedule(Seconds(tdir + (d++) * slot), &DirectSend, i, R.r, t);
                }
        t = tdir + (d + 2) * slot;
    }
    Simulator::Stop(Seconds(t + 1.0));
    Simulator::Run();
    std::printf("%zu,%llu,%llu,%llu,%llu,%llu,%llu,%.4f\n",
                rounds.size(),
                (unsigned long long)generated,
                (unsigned long long)rxAtCh,
                (unsigned long long)droppedMal,
                (unsigned long long)delivered,
                (unsigned long long)dataTx,
                (unsigned long long)macFail,
                delivered ? 1e3 * delaySum / delivered : 0.0);
    Simulator::Destroy();
    return 0;
}
