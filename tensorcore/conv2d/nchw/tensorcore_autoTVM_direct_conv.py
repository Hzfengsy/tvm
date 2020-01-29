import tvm
import numpy as np
from tvm import autotvm

TUNING = False
log_file = "conv2d_tensorcore_direct.log"

H = 14
W = 14
kernel = 3
pad = 1
stride = 1


def print_phase0(stmt):
    print(stmt)
    return stmt


def intrin_wmma_load_matrix_A(strides_dst, strides_from, shape, layout):
    wmma_m, wmma_n, wmma_k = shape

    A = tvm.placeholder((wmma_m, wmma_k, 1, 1), name='A', dtype='float16')
    BA = tvm.decl_buffer(A.shape, A.dtype,
                         scope='shared', strides=strides_from,
                         data_alignment=32, offset_factor=8)
    C = tvm.compute(A.shape, lambda *i: A(*i), name='C')
    BC = tvm.decl_buffer(C.shape, C.dtype,
                         scope="wmma.matrix_a", strides=strides_dst,
                         data_alignment=32, offset_factor=8)

    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        row = wmma_m * wmma_k
        warp_index = BC.elem_offset // row + BC.elem_offset % row // wmma_k
        ib.emit(tvm.call_intrin('handle', 'tvm_load_matrix_sync',
                                BC.data, wmma_m, wmma_n, wmma_k, warp_index,
                                BA.access_ptr('r'), strides_from[0], layout))
        return ib.get()

    return tvm.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_load_matrix_W(strides_dst, strides_from, shape, layout):
    wmma_m, wmma_n, wmma_k = shape

    A = tvm.placeholder((wmma_n, wmma_k, 1, 1), name='A', dtype='float16')
    BA = tvm.decl_buffer(A.shape, A.dtype,
                         scope='shared', strides=strides_from,
                         data_alignment=32, offset_factor=8)
    C = tvm.compute(A.shape, lambda *i: A(*i), name='C')
    BC = tvm.decl_buffer(C.shape, C.dtype,
                         scope="wmma.matrix_b", strides=strides_dst,
                         data_alignment=32, offset_factor=8)

    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        row = wmma_n * wmma_k
        warp_index = BC.elem_offset // row + BC.elem_offset % row // wmma_n
        ib.emit(tvm.call_intrin('handle', 'tvm_load_matrix_sync',
                                BC.data, wmma_m, wmma_n, wmma_k, warp_index,
                                BA.access_ptr('r'), strides_from[0], layout))
        return ib.get()

    return tvm.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_gemm(strides_A, strides_W, strides_Conv, shape):
    wmma_m, wmma_n, wmma_k = shape
    A = tvm.placeholder((wmma_m, wmma_k, 1, 1), name='A', dtype='float16')
    B = tvm.placeholder((wmma_n, wmma_k, 1, 1), name='B', dtype='float16')
    k = tvm.reduce_axis((0, wmma_k), name="k")
    C = tvm.compute((wmma_m, wmma_n, 1, 1), lambda i, j, t0, t1: tvm.sum(
        A[i, k, t0, t1] * B[j, k, t0, t1], axis=k), name='C')
    BA = tvm.decl_buffer(A.shape, A.dtype, name='BA',
                         scope='wmma.matrix_a', data_alignment=32,
                         offset_factor=8, strides=strides_A)
    BB = tvm.decl_buffer(B.shape, B.dtype, name='BB',
                         scope='wmma.matrix_b', data_alignment=32,
                         offset_factor=8, strides=strides_W)
    BC = tvm.decl_buffer(C.shape, C.dtype, name='BC',
                         scope='wmma.accumulator', data_alignment=32,
                         offset_factor=8, strides=strides_Conv)

    def intrin_func(ins, outs):
        BA, BB = ins
        BC, = outs

        def warp_idnex(offset, row, col):
            row = row * col
            return offset // row + offset % row // col

        warp_index_A = warp_idnex(BA.elem_offset, wmma_m, wmma_k)
        warp_index_B = warp_idnex(BB.elem_offset, wmma_k, wmma_n)
        warp_index_C = warp_idnex(BC.elem_offset, wmma_m, wmma_n)

        def init():
            ib = tvm.ir_builder.create()
            ib.emit(tvm.call_intrin('handle', 'tvm_fill_fragment', BC.data, wmma_m, wmma_n, wmma_k,
                                    warp_index_C, 0.0))
            return ib.get()

        def update():
            ib = tvm.ir_builder.create()
            ib.emit(tvm.call_intrin('handle', 'tvm_mma_sync',
                                    BC.data, warp_index_C,
                                    BA.data, warp_index_A,
                                    BB.data, warp_index_B,
                                    BC.data, warp_index_C))
            return ib.get()

        return update(), init(), update()

    return tvm.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})


def intrin_wmma_store_matrix(strides_dst, strides_from, shape):
    wmma_m, wmma_n, wmma_k = shape
    A = tvm.placeholder((wmma_m, wmma_n, 1, 1), name='A', dtype='float16')
    BA = tvm.decl_buffer(A.shape, A.dtype,
                         scope='wmma.accumulator',
                         strides=strides_from, data_alignment=32,
                         offset_factor=8)
    C = tvm.compute(A.shape, lambda *i: A(*i), name='C')
    BC = tvm.decl_buffer(C.shape, C.dtype,
                         scope='shared', strides=strides_dst,
                         data_alignment=32, offset_factor=8)

    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        row = wmma_m * wmma_n
        warp_index = BA.elem_offset // row + BA.elem_offset % row // wmma_n
        ib.emit(tvm.call_intrin('handle', 'tvm_store_matrix_sync',
                                BA.data, wmma_m, wmma_n, wmma_k, warp_index,
                                BC.access_ptr('w'), strides_dst[0], 'row_major'))
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

    # Input feature map: (N, C, H, W)
    data_shape = (batch_size, in_channels, height, width)
    # Kernel: (OC, IC, H, W)
    kernel_shape = (out_channels, in_channels, kernel_h, kernel_w)

    # Output feature map: (N, H, W, OC)
    output_shape = (batch_size, out_channels, height, width)

    kh = tvm.reduce_axis((0, kernel_h), name='kh')
    kw = tvm.reduce_axis((0, kernel_w), name='kw')
    ic = tvm.reduce_axis((0, in_channels), name='ic')

    # Algorithm
    A = tvm.placeholder(data_shape, name='A', dtype="float16")
    W = tvm.placeholder(kernel_shape, name='W', dtype="float16")
    Apad = tvm.compute((batch_size, in_channels, height + 2 * pad_h, width + 2 * pad_w),
                       lambda n, c, h, w: tvm.if_then_else(
                           tvm.all(h >= pad_h, h - pad_h < height,
                                   w >= pad_w, w - pad_w < width),
                           A[n, c, h - pad_h, w - pad_w], tvm.const(0, "float16")),
                       name='Apad')
    Conv = tvm.compute(output_shape, lambda n, c, h, w: tvm.sum(
        Apad[n, ic, h * stride_h + kh, w * stride_w + kw] * W[c, ic, kh, kw], axis=[kh, kw, ic]),
                       name="Conv")

    s = tvm.create_schedule(Conv.op)
    s[Apad].compute_inline()

    # Designate the memory hierarchy
    AS = s.cache_read(Apad, 'shared', [Conv])
    WS = s.cache_read(W, 'shared', [Conv])
    AF = s.cache_read(AS, 'wmma.matrix_a', [Conv])
    WF = s.cache_read(WS, 'wmma.matrix_b', [Conv])
    ConvF = s.cache_write(Conv, 'wmma.accumulator')
    ConvL = s.cache_read(ConvF, 'shared', [Conv])

    # Define tiling sizes
    cfg = autotvm.get_config()
    cfg.define_knob("block_row_warps", [1, 2, 4])
    cfg.define_knob("block_col_warps", [1, 2, 4])
    cfg.define_knob("warp_row_tiles", [1, 2, 4])
    cfg.define_knob("warp_col_tiles", [1, 2, 4])
    cfg.define_knob("chunk", [1, 2, 4, 8])
    cfg.define_knob("offset", [0, 8, 16])
    warp_size = 32

    block_row_warps = cfg["block_row_warps"].val
    block_col_warps = cfg["block_col_warps"].val
    warp_row_tiles = cfg["warp_row_tiles"].val
    warp_col_tiles = cfg["warp_col_tiles"].val
    chunk = cfg["chunk"].val
    offset = cfg["offset"].val

    block_row_warps = 2
    block_col_warps = 4
    warp_row_tiles = 2
    warp_col_tiles = 1
    chunk = 4
    offset = 0

    block_x = tvm.thread_axis('blockIdx.x')
    block_y = tvm.thread_axis('blockIdx.y')
    block_z = tvm.thread_axis('blockIdx.z')
    thread_x = tvm.thread_axis('threadIdx.x')
    thread_y = tvm.thread_axis('threadIdx.y')
    thread_z = tvm.thread_axis('threadIdx.z')

    def get_strides(extents):
        return [np.prod(extents[i:]).tolist() for i in range(len(extents))]

    AS_align = chunk * wmma_k + offset
    WS_align = chunk * wmma_k + offset
    AS_strides = get_strides([AS_align, 1, 1, 1])
    AL_strides = get_strides([wmma_k, 1, 1, 1])
    WS_strides = get_strides([WS_align, 1, 1, 1])
    WL_strides = get_strides([wmma_k, 1, 1, 1])
    CL_strides = get_strides([wmma_n * warp_col_tiles, 1, 1, 1])
    CS_strides = get_strides([wmma_n * warp_col_tiles * block_col_warps, 1, 1, 1])

    nc, oc, hc, wc = Conv.op.axis
    block_k = s[Conv].fuse(hc, wc)
    s[Conv].bind(block_k, block_z)
    nc, nnc = s[Conv].split(nc, factor=wmma_m)
    nc, nci = s[Conv].split(nc, factor=warp_row_tiles)
    block_i, nc = s[Conv].split(nc, factor=block_row_warps)
    oc, ooc = s[Conv].split(oc, factor=wmma_n)
    oc, oci = s[Conv].split(oc, factor=warp_col_tiles)
    block_j, oc = s[Conv].split(oc, factor=block_col_warps)
    s[Conv].reorder(block_k, block_i, block_j, nc, oc, nci, oci, nnc, ooc)
    t = s[Conv].fuse(nnc, ooc)
    to, ti = s[Conv].split(t, factor=warp_size)
    s[Conv].bind(block_i, block_x)
    s[Conv].bind(block_j, block_y)
    s[Conv].bind(nc, thread_y)
    s[Conv].bind(oc, thread_z)
    s[Conv].bind(ti, thread_x)

    s[ConvL].compute_at(s[Conv], block_j)
    n, o, h, w = ConvL.op.axis
    n, nnl = s[ConvL].split(n, factor=wmma_m)
    o, ool = s[ConvL].split(o, factor=wmma_n)
    no, ni = s[ConvL].split(n, factor=warp_row_tiles)
    oo, oi = s[ConvL].split(o, factor=warp_col_tiles)
    s[ConvL].reorder(no, oo, ni, oi, nnl, ool)
    s[ConvL].bind(no, thread_y)
    s[ConvL].bind(oo, thread_z)

    s[ConvF].compute_at(s[ConvL], oo)
    n, o, h, w = ConvF.op.axis
    n, nnf = s[ConvF].split(n, factor=wmma_m)
    o, oof = s[ConvF].split(o, factor=wmma_n)
    ic, ii = s[ConvF].split(ic, factor=wmma_k)
    ko, ki = s[ConvF].split(ic, factor=chunk)
    s[ConvF].reorder(kh, kw, ko, ki, n, o, nnf, oof, ii)

    s[AF].compute_at(s[ConvF], ki)
    s[WF].compute_at(s[ConvF], ki)

    s[WS].compute_at(s[ConvF], ko)
    s[AS].compute_at(s[ConvF], ko)

    n, i, h, w = AS.op.axis
    s[AS].reorder(h, w, n, i)
    s[AS].storage_align(h, AS_align - 1, AS_align)
    t = s[AS].fuse(n, i)
    tx, xo = s[AS].split(t, nparts=block_row_warps)
    ty, yo = s[AS].split(xo, nparts=block_col_warps)
    to, ti = s[AS].split(yo, factor=warp_size)
    s[AS].bind(tx, thread_y)
    s[AS].bind(ty, thread_z)
    s[AS].bind(ti, thread_x)

    o, ic, kh, kw = WS.op.axis
    t = s[WS].fuse(ic, o)
    s[WS].storage_align(o, WS_align - 1, WS_align)
    tx, xo = s[WS].split(t, nparts=block_row_warps)
    ty, yo = s[WS].split(xo, nparts=block_col_warps)
    to, ti = s[WS].split(yo, factor=warp_size)
    s[WS].bind(tx, thread_y)
    s[WS].bind(ty, thread_z)
    s[WS].bind(ti, thread_x)

    n, i, h, w = AF.op.axis
    n, nn = s[AF].split(n, factor=wmma_m)
    i, ii = s[AF].split(i, factor=wmma_k)
    s[AF].reorder(n, i, nn, ii)

    o, i, kh, kw = WF.op.axis
    i, ii = s[WF].split(i, factor=wmma_k)
    o, oo = s[WF].split(o, factor=wmma_n)
    s[WF].reorder(o, i, oo, ii)

    shape = (wmma_m, wmma_n, wmma_k)
    s[AF].tensorize(nn, intrin_wmma_load_matrix_A(AL_strides, AS_strides, shape, "row_major"))
    s[WF].tensorize(oo, intrin_wmma_load_matrix_W(WL_strides, WS_strides, shape, "col_major"))
    s[ConvL].tensorize(nnl, intrin_wmma_store_matrix(CS_strides, CL_strides, shape))
    s[ConvF].tensorize(nnf, intrin_wmma_gemm(AL_strides, WL_strides, CL_strides, shape))

    return s, [A, W, Conv]


def auto_tuning_task(B, IC, OC, M, K, N):
    if TUNING:
        task = autotvm.task.create(conv2d_with_tensorcore, args=(
            B, H, W, IC, OC, kernel, kernel, pad, pad, stride, stride, M, K, N),
                                   target='cuda')

        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4)
        )

        tuner = autotvm.tuner.GridSearchTuner(task)
        tuner.tune(n_trial=729,
                   measure_option=measure_option,
                   callbacks=[autotvm.callback.log_to_file(log_file)])

    # apply history best from log file
    with autotvm.apply_history_best(log_file):
        with tvm.target.create("cuda"):
            s, arg_bufs = conv2d_with_tensorcore(B, H, W, IC, OC, kernel, kernel,
                                                 pad, pad, stride, stride, M, K, N)
            print(tvm.lower(s, arg_bufs, simple_mode=True))
            func = tvm.build(s, arg_bufs)
            print(func.imported_modules[0].get_source())

    ctx = tvm.gpu()
    a_np = np.random.uniform(size=(B, IC, H, W)).astype("float16")
    w_np = np.random.uniform(size=(OC, IC, kernel, kernel)).astype("float16")
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    c = tvm.nd.array(np.zeros((B, OC, H, W), dtype="float16"), ctx)

    evaluator = func.time_evaluator(func.entry_name, ctx, number=1000)
    print(f'B={B}, OC={OC}, shape={M}_{N}: {evaluator(a, w, c).mean * 1e3} ms')


if __name__ == '__main__':
    # for B in [32, 64, 128, 256]:
    #     for IC in [64, 256]:
    #         for OC in [64, 256]:
    #             for (M, K, N) in [(16, 16, 16), (32, 16, 8), (8, 16, 32)]:
    #                 if OC % N != 0:
    #                     continue
    #                 auto_tuning_task(B, OC, M, K, N)
    auto_tuning_task(32, 256, 256, 8, 16, 32)
