import numpy as np
import matplotlib.pyplot as plt
from .report import parse_xml

"""
Ref: https://www.xilinx.com/support/documentation/data_sheets/ds190-Zynq-7000-Overview.pdf
XC7Z020 276 GMACs
https://www.xilinx.com/support/documentation/boards_and_kits/zc706/ug954-zc706-eval-board-xc7z045-ap-soc.pdf
DDR3 SODIMM Memory (PL)
Datapath width: 64 bits
Data rate: Up to 1,600 MT/s
"""

class Profiler():

    def __init__(self):
        self.perf = {}
        self.clock = 100 * 1000 * 1000 # MHz = 10e8 cycles/s 10ns/cycle
        self.bandwidth_roof = 1600 * 1000 * 1000 * 8 # B/s
        self.compute_roof = 276 * 1000 * 1000 * 1000 # FLOP/s
        self.ridge_point = self.compute_roof / self.bandwidth_roof
        self.initialize_perf()

    def clear(self):
        self.perf = {}
        self.initialize_perf()

    def initialize_perf(self):
        self.perf["store"] = []
        self.perf["load"] = []
        self.perf["op"] = []
        self.perf["ai"] = []
        self.perf["perf"] = []

    def get_info(self,*vcnt):
        """
        PackedFunc
        Do not call this function directly!
        """
        self.perf["store"].append(vcnt[0])
        self.perf["load"].append(vcnt[1])
        self.perf["op"].append(vcnt[2])
        self.perf["ai"].append(float(self.perf["op"][-1]) / float(self.perf["store"][-1] + self.perf["load"][-1])) # arithmetic intensity
        print("Store + Load: {} B + {} B = {} B".format(self.perf["store"][-1],self.perf["load"][-1],self.perf["store"][-1] + self.perf["load"][-1]))
        print("# of ops: {} GFLOS".format(self.perf["op"][-1] / 10**9))
        print("Arithmetic density: {} FLOPs/Byte".format(self.perf["ai"][-1]))
        print("Ridge point: {} FLOPs/Byte".format(self.ridge_point))
        print("I/O bandwidth roof: {:.2f} GB/s".format(self.bandwidth_roof/10**9))
        print("Compute roof: {:.2f} GFLOP/s".format(self.compute_roof/10**9))
        if self.ridge_point > self.perf["ai"][-1]:
            print("Memory bound!")
        else:
            print("Computation bound!")

    def profile_report(self,f=None,target=None):
        if f != None and target != None:
            report = f.report(target)
        else:
            report = parse_xml("project")
        latency = report["PerformanceEstimates"]["SummaryOfOverallLatency"]["Best-caseLatency"]
        self.perf["perf"].append(self.perf["op"][-1] / int(latency) * self.clock) # FLOP/cycles * cycles/s -> FLOP/s
        print("Real performance: {} GFLOP/s".format(self.perf["perf"][-1]/(10**9)))

    def roofline(self,log_plot=True,filename="roofline.png"):
        """
        Ref: Samuel Williams, Andrew Waterman, and David Patterson,
            Roofline: An Insightful Visual Performance Model for
            Floating-Point Programs and Multicore Architectures
        """
        max_x = 100 if log_plot else 60
        sample_interval = 1
        x = np.arange(0,max_x,sample_interval).astype(np.int64)
        y = np.minimum(x * self.bandwidth_roof, np.ones(x.shape) * self.compute_roof)
        fig = plt.figure()
        ax = fig.gca()
        if log_plot:
            plt.xscale("log")
            plt.yscale("log")
        plt.plot(x, y)
        plt.plot(self.perf["ai"], np.array(self.perf["ai"])*self.bandwidth_roof,"x",color="orange",label="Attainable Performance")
        plt.plot(self.perf["ai"], self.perf["perf"],"o",color="red",label="Real Performance")
        for i in range(len(self.perf["ai"])):
            plt.vlines(x=self.perf["ai"][i], ymin=0, ymax=min(self.perf["ai"][i] * self.bandwidth_roof, self.compute_roof), linestyle="--",color="orange")
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        plt.title("Roofline Model")
        plt.xlabel("Arithmetic density (FLOPs/Byte)")
        plt.ylabel("Performance (FLOPs/sec)")
        plt.legend(loc=0)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()