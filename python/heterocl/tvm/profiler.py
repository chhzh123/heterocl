import numpy as np
import matplotlib.pyplot as plt

clock = 100 * 1000 * 1000 # MHz = 10e8 cycles/s 10ns/cycle

def cal_roof():
    """
    Ref: https://www.xilinx.com/support/documentation/data_sheets/ds190-Zynq-7000-Overview.pdf
    XC7Z020 276 GMACs
    https://www.xilinx.com/support/documentation/boards_and_kits/zc706/ug954-zc706-eval-board-xc7z045-ap-soc.pdf
    DDR3 SODIMM Memory (PL)
    Datapath width: 64 bits
    Data rate: Up to 1,600 MT/s
    """
    bandwidth_roof = 1600 * 1000 * 1000 * 8 # B/s
    compute_roof = 276 * 1000 * 1000 * 1000 # FLOP/s
    return bandwidth_roof, compute_roof

def roofline(*vcnt,log_plot=True,filename="roofline.png"):
    """
    Ref: Samuel Williams, Andrew Waterman, and David Patterson,
        Roofline: An Insightful Visual Performance Model for
        Floating-Point Programs and Multicore Architectures
    """
    store, load, op, loop = vcnt
    access = (store+load)/ (10**6) #(1024 * 1024)
    op = op / (10**6)
    density = float(op) / float(access)
    performance = op/op * clock # FLOP/cycles * cycles/s -> FLOP/s
    print("Store + Load: {} B + {} B = {} MB".format(store,load,access))
    print("# op: {} GFLOS".format(op / 1000))
    print("Arithmetic density: {} FLOPs/Byte".format(density))
    max_x = 100 if log_plot else 60
    sample_width = 1
    br, cr = cal_roof()
    x = np.arange(0,max_x,sample_width).astype(np.int64)
    y = np.minimum(x * br, np.ones(x.shape) * cr)
    ridge_point = cr / br
    print("Ridge point: {} FLOPs/Byte".format(ridge_point))
    if ridge_point > density:
        print("Memory bound!")
    else:
        print("Computation bound!")
    # plot roofline model
    fig = plt.figure()
    ax = fig.gca()
    if log_plot:
        plt.xscale("log")
        plt.yscale("log")
    plt.plot(x, y)
    plt.plot([density],[min(density*br, cr)],"x",color="orange",label="Attainable Performance")
    plt.plot([density],[performance],"o",color="red",label="Real Performance")
    plt.vlines(x=density, ymin=0, ymax=min(density*br, cr), linestyle="--",color="orange")
    # plt.text(ridge_point, cr+10**9, "Peak performance: {:.1f}GFLOP/s".format(cr/(10**9)))
    # if log_plot:
    #     angle = np.arctan(np.log10((ridge_point-20)*br) / np.log10(ridge_point-20)) / np.pi * 180
    # else:
    #     angle_data = np.rad2deg(np.arctan2(y[1]-y[0], x[1]-x[0]))
    #     angle = ax.transData.transform_angles(np.array((angle_data,)), 
    #                                           np.array([x[0], y[0]]).reshape((1, 2)))[0]
    # plt.text(ridge_point-20, (ridge_point-20)*br, "I/O bandwidth roof: {:.1f}GB/s".format(br/(10**9)),rotation=angle)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.title("Roofline Model")
    plt.xlabel("Arithmetic density (FLOPs/Byte)")
    plt.ylabel("Performance (GFLOPs/sec)")
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()