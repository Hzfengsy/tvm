import tvm
import numpy as np
from tvm import autotvm

TUNING = True
log_file = "conv2d_tensorcore.log"

H = 14
W = 14
IC = 256
kernel = 3
pad = 1
stride = 1


def intrin_wmma_load_matrix_A(scope, M, K, N):
    factor = M * K
    A = tvm.placeholder((M, K), name='A', dtype='float16')
    BA = tvm.decl_buffer(A.shape, A.dtype, scope='shared', data_alignment=32, offset_factor=factor)
    C = tvm.compute((M, K), lambda i, j: A[i, j], name='C')
    BC = tvm.decl_buffer(C.shape, C.dtype, scope=scope, data_alignment=32, offset_factor=factor)

    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(tvm.call_intrin('handle', 'tvm_load_matrix_sync',
            BC.data, M, N, K, BC.elem_offset // factor,
            BA.access_ptr('r'), K, 'row_major'))
        return ib.get()

    return tvm.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_load_matrix_W(scope, M, K, N):
    factor = K * N
    A = tvm.placeholder((K, N), name='A', dtype='float16')
    BA = tvm.decl_buffer(A.shape, A.dtype, scope='shared', data_alignment=32, offset_factor=factor)
    C = tvm.compute((K, N), lambda i, j: A[i, j], name='C')
    BC = tvm.decl_buffer(C.shape, C.dtype, scope=scope, data_alignment=32, offset_factor=factor)

    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(tvm.call_intrin('handle', 'tvm_load_matrix_sync',
            BC.data, M, N, K, BC.elem_offset // factor,
            BA.access_ptr('r'), N, 'row_major'))
        return ib.get()

    return tvm.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_gemm(M, K, N):
    A = tvm.placeholder((M, K), name='A', dtype='float16')
    B = tvm.placeholder((K, N), name='B', dtype='float16')
    k = tvm.reduce_axis((0, K), name="k")
    C = tvm.compute((M, N),
        lambda ii, jj:
        tvm.sum(A[ii, k].astype('float') * B[k, jj].astype('float'), axis=k),
        name='C')
    BA = tvm.decl_buffer(A.shape, A.dtype, name='BA', scope='wmma.matrix_a', data_alignment=32,
        offset_factor=M * K)
    BB = tvm.decl_buffer(B.shape, B.dtype, name='BB', scope='wmma.matrix_b', data_alignment=32,
        offset_factor=K * N)
    BC = tvm.decl_buffer(C.shape, C.dtype, name='BC', scope='wmma.accumulator', data_alignment=32,
        offset_factor=M * N)

    def intrin_func(ins, outs):
        BA, BB = ins
        BC, = outs

        def init():
            ib = tvm.ir_builder.create()
            ib.emit(tvm.call_intrin('handle', 'tvm_fill_fragment', BC.data, M, N, K,
                BC.elem_offset // (M * N), 0.0))
            return ib.get()

        def update():
            ib = tvm.ir_builder.create()
            ib.emit(tvm.call_intrin('handle', 'tvm_mma_sync',
                BC.data, BC.elem_offset // (M * N),
                BA.data, BA.elem_offset // (M * K),
                BB.data, BB.elem_offset // (K * N),
                BC.data, BC.elem_offset // (M * N)))
            return ib.get()

        return update(), init(), update()

    return tvm.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})


def intrin_wmma_store_matrix(M, K, N):
    A = tvm.placeholder((M, N), name='A', dtype='float32')
    BA = tvm.decl_buffer(A.shape, A.dtype, scope='wmma.accumulator', data_alignment=32,
        offset_factor=(M * N))
    C = tvm.compute((M, N), lambda i, j: A[i, j], name='C')
    BC = tvm.decl_buffer(C.shape, C.dtype, scope='global', data_alignment=32, offset_factor=(M * N))

    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        ib.emit(tvm.call_intrin('handle', 'tvm_store_matrix_sync',
            BA.data, M, N, K, BA.elem_offset // (M * N),
            BC.access_ptr('w'), N, 'row_major'))
        return ib.get()

    return tvm.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


# The sizes of inputs and filters
@autotvm.template
def conv2d_with_tensorcore(batch_size, height, width, in_channels, out_channels,
                           kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
                           wmma_m, wmma_k, wmma_n):
    assert (batch_size % wmma_m == 0)
    assert (in_channels % wmma_k == 0)
    assert (out_channels % wmma_n == 0)

    # Input feature map: (N, H, W, IC)
    data_shape = (batch_size,
                  height,
                  width,
                  in_channels)
    # Kernel: (H, W, IC, OC)
    kernel_shape = (kernel_h,
                    kernel_w,
                    in_channels,
                    out_channels)
    # Output feature map: (N, H, W, OC)
    output_shape = (batch_size,
                    height,
                    width,
                    out_channels)

    # Reduction axes
    kh = tvm.reduce_axis((0, kernel_h), name='kh')
    kw = tvm.reduce_axis((0, kernel_w), name='kw')
    ic = tvm.reduce_axis((0, in_channels // wmma_k), name='ic')
    ii = tvm.reduce_axis((0, wmma_k), name='ii')

    # Algorithm
    A = tvm.placeholder(data_shape, name='A', dtype="float16")
    W = tvm.placeholder(kernel_shape, name='W', dtype="float16")
    Apad = tvm.compute(
        (batch_size // wmma_m, height + 2 * pad_h, width + 2 * pad_w, in_channels // wmma_k, wmma_m,
         wmma_k),
        lambda n, h, w, i, nn, ii: tvm.if_then_else(
            tvm.all(h >= pad_h, h - pad_h < height,
                    w >= pad_w, w - pad_w < width),
            A[n, h - pad_h, w - pad_w, i, nn, ii], tvm.const(0., "float16")),
        name='Apad')
    W_shared = tvm.compute(
        (batch_size // wmma_m, height + 2 * pad_h, width + 2 * pad_w, in_channels // wmma_k, wmma_m,
         wmma_k),
        lambda n, h, w, i, nn, ii: tvm.if_then_else(
            tvm.all(h >= pad_h, h - pad_h < height,
                    w >= pad_w, w - pad_w < width),
            A[n, h - pad_h, w - pad_w, i, nn, ii], tvm.const(0., "float16")),
        name='Apad')
    Conv = tvm.compute(output_shape,
        lambda n, h, w, o, nn, oo: tvm.sum(
            Apad[n, h * stride_h + kh, w * stride_w + kw, ic, nn, ii].astype("float32") *
            W[kh, kw, ic, o, ii, oo].astype("float32"),
            axis=[ic, kh, kw, ii]),
        name="Conv")

    s = tvm.create_schedule(Conv.op)
    s[Apad].compute_inline()

    # Designate the memory hierarchy
    AS = s.cache_read(Apad, 'shared', [Conv])
    WS = s.cache_read(W, 'shared', [Conv])
    AF = s.cache_read(AS, 'wmma.matrix_a', [Conv])
    WF = s.cache_read(WS, 'wmma.matrix_b', [Conv])
    ConvF = s.cache_write(Conv, 'wmma.accumulator')

    # Define tiling sizes
    cfg = autotvm.get_config()
    cfg.define_knob("block_row_warps", [1, 2, 4])
    cfg.define_knob("block_col_warps", [1, 2, 4])
    cfg.define_knob("warp_row_tiles", [1, 2, 4])
    cfg.define_knob("warp_col_tiles", [1, 2, 4])
    cfg.define_knob("chunk", [1, 2, 4])
    warp_size = 32

    block_x = tvm.thread_axis('blockIdx.x')
    block_y = tvm.thread_axis('blockIdx.y')
    block_z = tvm.thread_axis('blockIdx.z')
    thread_x = tvm.thread_axis('threadIdx.x')
    thread_y = tvm.thread_axis('threadIdx.y')
    thread_z = tvm.thread_axis('threadIdx.z')

    nc, hc, wc, oc, nnc, ooc = Conv.op.axis
    block_k = s[Conv].fuse(hc, wc)
    s[Conv].bind(block_k, block_z)
    nc, nci = s[Conv].split(nc, factor=cfg["warp_row_tiles"].val)
    block_i, nc = s[Conv].split(nc, factor=cfg["block_row_warps"].val)
    oc, oci = s[Conv].split(oc, factor=cfg["warp_col_tiles"].val)
    block_j, oc = s[Conv].split(oc, factor=cfg["block_col_warps"].val)
    s[Conv].reorder(block_k, block_i, block_j, nc, oc, nci, oci, nnc, ooc)
    s[Conv].bind(block_i, block_x)
    s[Conv].bind(block_j, block_y)

    s[Conv].bind(nc, thread_y)
    s[Conv].bind(oc, thread_z)
    s[ConvF].compute_at(s[Conv], oc)
    n, h, w, o, nnf, oof = ConvF.op.axis
    ko, ki = s[ConvF].split(ic, factor=2)
    s[ConvF].reorder(ko, kh, ki, kw, n, o, nnf, oof, ii)

    s[AF].compute_at(s[ConvF], kw)
    s[WF].compute_at(s[ConvF], kw)

    s[AS].compute_at(s[ConvF], kh)
    n, h, w, i, nn, ii = AS.op.axis
    tx, xo = s[AS].split(n, nparts=cfg["block_row_warps"].val)
    ty, yo = s[AS].split(xo, nparts=cfg["block_col_warps"].val)
    t = s[AS].fuse(nn, ii)
    to, ti = s[AS].split(t, factor=warp_size)
    s[AS].bind(tx, thread_y)
    s[AS].bind(ty, thread_z)
    s[AS].bind(ti, thread_x)

    # Schedule for W's share memory
    s[WS].compute_at(s[ConvF], kh)
    kh, kw, ic, o, ii, oo = WS.op.axis
    tx, xo = s[WS].split(o, factor=cfg["block_col_warps"].val)
    iy, ic = s[WS].split(ic, factor=cfg["chunk"].val)
    t = s[WS].fuse(ii, oo)
    to, ti = s[WS].split(t, factor=8)
    tn, to = s[WS].split(to, factor=32)
    s[WS].bind(ic, thread_y)
    s[WS].bind(xo, thread_z)
    s[WS].bind(to, thread_x)
    s[WS].vectorize(ti)

    s[AF].tensorize(AF.op.axis[-2], intrin_wmma_load_matrix_A('wmma.matrix_a', M, K, N))
    s[WF].tensorize(WF.op.axis[-2], intrin_wmma_load_matrix_W('wmma.matrix_b', M, K, N))
    s[Conv].tensorize(nnc, intrin_wmma_store_matrix(M, K, N))
    s[ConvF].tensorize(nnf, intrin_wmma_gemm(M, K, N))

    return s, [A, W, Conv]


def auto_tuning_task(B, OC, M, K, N):
    if TUNING:
        task = autotvm.task.create(conv2d_with_tensorcore,
              args=(B, H, W, IC, OC, kernel, kernel, pad, pad, stride, stride, M, K, N),
            target='cuda')

        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4)
        )

        tuner = autotvm.tuner.GridSearchTuner(task)
        tuner.tune(n_trial=243,
                   measure_option=measure_option,
                   callbacks=[autotvm.callback.log_to_file(log_file)])

    # apply history best from log file
    with autotvm.apply_history_best(log_file):
        with tvm.target.create("cuda"):
            s, arg_bufs = conv2d_with_tensorcore(B, H, W, IC, OC, kernel, kernel,
                pad, pad, stride, stride, M, K, N)
            func = tvm.build(s, arg_bufs)

    ctx = tvm.gpu(0)
    a_np = np.random.uniform(size=(B // M, H, W, IC // K, M, K)).astype("float16")
    w_np = np.random.uniform(size=(kernel, kernel, IC // K, OC // N, K, N)).astype("float16")
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    c = tvm.nd.array(np.zeros((B // M, H, W, OC // N, M, N), dtype="float32"), ctx)

    evaluator = func.time_evaluator(func.entry_name, ctx, number=1000)
    print(f'B={B}, OC={OC}, shape={M}_{N}: {evaluator(a, w, c).mean * 1e3} ms')


if __name__ == '__main__':
    for B in [32, 64, 128, 256]:
        for OC in [32, 64, 128, 256, 512, 1024]:
            for (M, K, N) in [(16, 16, 16), (32, 16, 8), (8, 16, 32)]:
                if OC % N != 0:
                    continue
                auto_tuning_task(B, OC, M, K, N)
