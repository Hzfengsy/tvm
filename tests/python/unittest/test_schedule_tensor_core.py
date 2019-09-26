# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# 'License'); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm
import numpy as np

def intrin_wmma_load_matrix(scope):
    n = 16
    A = tvm.placeholder((n, n), name='A', dtype='float16')
    BA = tvm.decl_buffer(A.shape, A.dtype, scope='shared', data_alignment=32, offset_factor=256)
    C = tvm.compute((n, n), lambda i, j: A[i, j], name='C')
    BC = tvm.decl_buffer(C.shape, C.dtype, scope=scope, data_alignment=32, offset_factor=256)

    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(tvm.call_intrin('handle', 'tvm_load_matrix_sync',
                                BC.data, BC.elem_offset / 256,
                                BA.access_ptr('r'), n))
        return ib.get()

    return tvm.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})

def intrin_wmma_gemm():
    n = 16
    A = tvm.placeholder((n, n), name='A', dtype='float16')
    B = tvm.placeholder((n, n), name='B', dtype='float16')
    k = tvm.reduce_axis((0, n), name="k")
    C = tvm.compute((n, n),
                     lambda ii, jj:
                     tvm.sum(A[ii, k].astype('float') * B[k, jj].astype('float'), axis=k),
                     name='C')
    BA = tvm.decl_buffer(A.shape, A.dtype, name='BA', scope='wmma.matrix_a', data_alignment=32, offset_factor=256)
    BB = tvm.decl_buffer(B.shape, B.dtype, name='BB', scope='wmma.matrix_b', data_alignment=32, offset_factor=256)
    BC = tvm.decl_buffer(C.shape, C.dtype, name='BC', scope='wmma.accumulator', data_alignment=32, offset_factor=256)

    def intrin_func(ins, outs):
        BA, BB = ins
        BC, = outs
        def init():
            ib = tvm.ir_builder.create()
            ib.emit(tvm.call_intrin('handle', 'tvm_fill_fragment', BC.data, BC.elem_offset / 256, 0.0))
            return ib.get()

        def update():
            ib = tvm.ir_builder.create()
            ib.emit(tvm.call_intrin('handle', 'tvm_mma_sync',
                                    BC.data, BC.elem_offset / 256,
                                    BA.data, BA.elem_offset / 256,
                                    BB.data, BB.elem_offset / 256,
                                    BC.data, BC.elem_offset / 256))
            return ib.get()
        return update(), init(), update()
    return tvm.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})

def intrin_wmma_store_matrix():
    n = 16
    A = tvm.placeholder((n, n), name='A', dtype='float32')
    BA = tvm.decl_buffer(A.shape, A.dtype, scope='wmma.accumulator', data_alignment=32, offset_factor=256)
    C = tvm.compute((n, n), lambda i, j: A[i, j], name='C')
    BC = tvm.decl_buffer(C.shape, C.dtype, scope='shared', data_alignment=32, offset_factor=256)

    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(tvm.call_intrin('handle', 'tvm_store_matrix_sync',
                                BA.data, BA.elem_offset / 256,
                                BC.access_ptr('w'), n, 'mem_row_major'))
        return ib.get()

    return tvm.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})

def test_tensor_core_float():
    n = 1024
    m, l = n, n
    nn, mm, ll = n // 16, m // 16, l // 16
    A = tvm.placeholder((n, l), name='A', dtype='float16')
    B = tvm.placeholder((l, m), name='B', dtype='float16')
    AS = tvm.compute((nn, ll, 16, 16), lambda i, j, ii, jj: A[i * 16 + ii, j * 16 + jj], name='A.shared')
    BS = tvm.compute((ll, nn, 16, 16), lambda i, j, ii, jj: B[i * 16 + ii, j * 16 + jj], name='B.shared')
    k1 = tvm.reduce_axis((0, ll), name='k1')
    k2 = tvm.reduce_axis((0, 16), name='k2')
    CF = tvm.compute((nn, mm, 16, 16),
                              lambda i, j, ii, jj:
                              tvm.sum(AS[i, k1, ii, k2].astype('float') * BS[k1, j, k2, jj].astype('float'), axis=[k1, k2]),
                              name='Fragment_C')
    C = tvm.compute((n, m), lambda ii, jj: CF[ii // 16, jj // 16, ii % 16, jj % 16], name='C')
    s = tvm.create_schedule(C.op)

    warp_size = 32
    kernel_size = 16
    block_row_warps = 2
    block_col_warps = 4
    warp_row_tiles = 2
    warp_col_tiles = 1
    chunk = 4

    block_x = tvm.thread_axis('blockIdx.x')
    block_y = tvm.thread_axis('blockIdx.y')
    thread_x = tvm.thread_axis('threadIdx.x')
    thread_y = tvm.thread_axis('threadIdx.y')
    thread_z = tvm.thread_axis('threadIdx.z')

    AF = s.cache_read(AS, 'wmma.matrix_a', [CF])
    BF = s.cache_read(BS, 'wmma.matrix_b', [CF])
    CS = s.cache_read(CF, 'shared', [C])
    s[AS].set_scope('shared')
    s[BS].set_scope('shared')
    s[CF].set_scope('wmma.accumulator')

    i, j = s[C].op.axis
    i, ii = s[C].split(i, factor=kernel_size * warp_row_tiles)
    block_i, i = s[C].split(i, factor=block_row_warps)
    j, jj = s[C].split(j, factor=kernel_size * warp_col_tiles)
    block_j, j = s[C].split(j, factor=block_col_warps)
    s[C].reorder(block_i, block_j, i, j, ii, jj)
    s[C].bind(block_i, block_x)
    s[C].bind(block_j, block_y)
    s[C].bind(i, thread_y)
    s[C].bind(j, thread_z)
    thread_i, _ = s[C].split(ii, nparts=warp_size)
    s[C].bind(thread_i, thread_x)

    s[CS].compute_at(s[C], j)
    s[CF].compute_at(s[C], j)
    xo, yo, xi, yi = CS.op.axis
    tx, xo = s[CS].split(xo, nparts=block_row_warps)
    ty, yo = s[CS].split(yo, nparts=block_col_warps)
    s[CS].bind(tx, thread_y)
    s[CS].bind(ty, thread_z)

    warp_i, warp_j, _i, _j = s[CF].op.axis
    k, _k = CF.op.reduce_axis
    ko, ki = s[CF].split(k, factor=chunk)
    s[CF].reorder(ko, ki, warp_i, warp_j, _i, _j, _k)

    s[AF].compute_at(s[CF], ki)
    s[BF].compute_at(s[CF], ki)

    s[AS].compute_at(s[CF], ko)
    xo, yo, xi, yi = AS.op.axis
    tx, xo = s[AS].split(xo, nparts=block_row_warps)
    ty, yo = s[AS].split(yo, nparts=block_col_warps)
    t = s[AS].fuse(xi, yi)
    to, ti = s[AS].split(t, nparts=warp_size)
    s[AS].bind(tx, thread_y)
    s[AS].bind(ty, thread_z)
    s[AS].bind(to, thread_x)

    s[BS].compute_at(s[CF], ko)
    xo, yo, xi, yi = BS.op.axis
    tx, xo = s[BS].split(xo, nparts=block_row_warps)
    ty, yo = s[BS].split(yo, nparts=block_col_warps)
    t = s[BS].fuse(xi, yi)
    to, ti = s[BS].split(t, nparts=warp_size)
    s[BS].bind(tx, thread_y)
    s[BS].bind(ty, thread_z)
    s[BS].bind(to, thread_x)

    s[AF].tensorize(AF.op.axis[-2], intrin_wmma_load_matrix('wmma.matrix_a'))
    s[BF].tensorize(BF.op.axis[-2], intrin_wmma_load_matrix('wmma.matrix_b'))
    s[CS].tensorize(CS.op.axis[-2], intrin_wmma_store_matrix())
    s[CF].tensorize(_i, intrin_wmma_gemm())
    func = tvm.build(s, [A, B, C], 'cuda')

    ctx = tvm.gpu(0)
    a_np = np.random.uniform(size=(n, n)).astype(A.dtype)
    b_np = np.random.uniform(size=(n, n)).astype(B.dtype)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(np.zeros((n, n), dtype=C.dtype), ctx)
    func(a, b, c)
    c_np = c.asnumpy()
    np.testing.assert_allclose(c_np, np.dot(a_np, b_np).astype(C.dtype), rtol=0.001, atol=0.001)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=5)
    print('gemm with tensor core: %f ms' % (evaluator(a, b, c).mean * 1e3))


if __name__ == '__main__':
    test_tensor_core_float()
