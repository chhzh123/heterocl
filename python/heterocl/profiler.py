import numpy as np
import matplotlib.pyplot as plt

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

    def get_info(self,*vcnt):
        self.perf["store"] = vcnt[0]
        self.perf["load"] = vcnt[1]
        self.perf["op"] = vcnt[2]
        self.perf["loop"] = vcnt[3]
        access = (self.perf["store"] + self.perf["load"])/ (10**6)
        op_ = self.perf["op"] / (10**6)
        self.perf["ai"] = float(op_) / float(access) # arithmetic intensity
        self.perf["perf"] = op_/op_ * self.clock # FLOP/cycles * cycles/s -> FLOP/s
        print("Store + Load: {} B + {} B = {} MB".format(self.perf["store"],self.perf["load"],access))
        print("# of ops: {} GFLOS".format(op_ / 1000))
        print("Arithmetic density: {} FLOPs/Byte".format(self.perf["ai"]))
        print("Ridge point: {} FLOPs/Byte".format(self.ridge_point))
        if self.ridge_point > self.perf["ai"]:
            print("Memory bound!")
        else:
            print("Computation bound!")

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
        # plot roofline model
        fig = plt.figure()
        ax = fig.gca()
        if log_plot:
            plt.xscale("log")
            plt.yscale("log")
        plt.plot(x, y)
        plt.plot([self.perf["ai"]],[min(self.perf["ai"]*self.bandwidth_roof, self.compute_roof)],"x",color="orange",label="Attainable Performance")
        plt.plot([self.perf["ai"]],[self.perf["perf"]],"o",color="red",label="Real Performance")
        plt.vlines(x=self.perf["ai"], ymin=0, ymax=min(self.perf["ai"] * self.bandwidth_roof, self.compute_roof), linestyle="--",color="orange")
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        plt.title("Roofline Model")
        plt.xlabel("Arithmetic density (FLOPs/Byte)")
        plt.ylabel("Performance (GFLOPs/sec)")
        plt.legend(loc=0)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()