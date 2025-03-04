import numpy as np
import math
import tensorflow as tf
from numba import njit

@njit
def int2bin_numba(num, width):
    result = np.empty(width, dtype=np.int32)
    for i in range(width):
        result[i] = (num >> (width - 1 - i)) & 1
    return result

@njit
def bmcalc_numba(llr_in, conv_n, num_syms, no):
    batch = llr_in.shape[0]
    op_bits = np.empty((no, conv_n), dtype=np.int32)
    for op in range(no):
        op_bits[op, :] = int2bin_numba(op, conv_n)

    bm = np.empty((batch, no, num_syms), dtype=np.float32)
    for b in range(batch):
        for i in range(no):
            for j in range(num_syms):
                s = 0.0
                for k in range(conv_n):
                    sign = 1.0 - 2.0 * op_bits[i, k]
                    s += 0.5 * llr_in[b, j * conv_n + k] * sign
                bm[b, i, j] = s
    return bm

@njit
def update_fwd_numba(alph_init, bm_mat, llr, ipst_ip_idx, op_by_fromnode, num_syms):
    batch, Ns = alph_init.shape
    ni = ipst_ip_idx.shape[1]
    alph = np.empty((num_syms + 1, batch, Ns), dtype=np.float32)
    for b in range(batch):
        for s in range(Ns):
            alph[0, b, s] = alph_init[b, s]
    for t in range(num_syms):
        for b in range(batch):
            L = 0.5 * llr[b, t]
            for s in range(Ns):
                max_val = -1e30  
                for i in range(ni):
                    prev_state = ipst_ip_idx[s, i, 0]
                    op_val = op_by_fromnode[prev_state, i]
                    val = alph[t, b, prev_state] + L * (1.0 - 2.0 * i) + bm_mat[b, op_val, t]
                    if val > max_val:
                        max_val = val
                alph[t + 1, b, s] = max_val
    return alph

@njit
def update_bwd_numba(beta_init, bm_mat, llr, alph, op_by_fromnode, to_nodes, num_syms):
    batch, Ns = beta_init.shape
    ni = op_by_fromnode.shape[1]
    beta = np.empty((num_syms + 1, batch, Ns), dtype=np.float32)
    for b in range(batch):
        for s in range(Ns):
            beta[num_syms, b, s] = beta_init[b, s]
    for t in range(num_syms - 1, -1, -1):
        for b in range(batch):
            for s in range(Ns):
                max_val = -1e30
                for i in range(ni):
                    ns = to_nodes[s, i]
                    val = 0.5 * llr[b, t] * (1.0 - 2.0 * i) + bm_mat[b, op_by_fromnode[s, i], t] + beta[t + 1, b, ns]
                    if val > max_val:
                        max_val = val
                beta[t, b, s] = max_val
    llr_op = np.empty((batch, num_syms), dtype=np.float32)
    for t in range(num_syms):
        for b in range(batch):
            max0 = -1e30
            max1 = -1e30
            for s in range(Ns):
                val0 = alph[t, b, s] + 0.5 * llr[b, t] * 1.0 + bm_mat[b, op_by_fromnode[s, 0], t] + beta[t + 1, b, to_nodes[s, 0]]
                if val0 > max0:
                    max0 = val0
                val1 = alph[t, b, s] + 0.5 * llr[b, t] * (-1.0) + bm_mat[b, op_by_fromnode[s, 1], t] + beta[t + 1, b, to_nodes[s, 1]]
                if val1 > max1:
                    max1 = val1
            llr_op[b, t] = max0 - max1
    return llr_op

class Trellis:
    def __init__(self, gen_poly, rsc=True):
        self.rsc = rsc
        self.gen_poly = gen_poly
        self.constraint_length = len(self.gen_poly[0])
        self.conv_k = 1
        self.conv_n = len(self.gen_poly)
        self.ni = 2 ** self.conv_k
        self.ns = 2 ** (self.constraint_length - 1)
        self._mu = len(gen_poly[0]) - 1

        if self.rsc:
            self.fb_poly = [int(x) for x in self.gen_poly[0]]
            assert self.fb_poly[0] == 1

        self.to_nodes = np.full((self.ns, self.ni), -1, dtype=np.int32)
        self.from_nodes = np.full((self.ns, self.ni), -1, dtype=np.int32)
        self.op_mat = np.full((self.ns, self.ns), -1, dtype=np.int32)
        self.ip_by_tonode = np.full((self.ns, self.ni), -1, dtype=np.int32)
        self.op_by_tonode = np.full((self.ns, self.ni), -1, dtype=np.int32)
        self.op_by_fromnode = np.full((self.ns, self.ni), -1, dtype=np.int32)
        self._generate_transitions()

    def _binary_matmul(self, st):
        op = np.zeros(self.conv_n, dtype=np.int32)
        for i, poly in enumerate(self.gen_poly):
            op_int = sum(int(char) * int(poly[idx]) for idx, char in enumerate(st))
            op[i] = int2bin(op_int % 2, 1)[0]
        return op

    def _binary_vecmul(self, v1, v2):
        op_int = sum(x * int(v2[idx]) for idx, x in enumerate(v1))
        return int2bin(op_int, 1)[0]

    def _generate_transitions(self):
        to_nodes = np.full((self.ns, self.ni), -1, int)
        from_nodes = np.full((self.ns, self.ni), -1, int)
        op_mat = np.full((self.ns, self.ns), -1, int)
        ip_by_tonode =  np.full((self.ns, self.ni), -1, int)
        op_by_tonode =  np.full((self.ns, self.ni), -1, int)
        op_by_fromnode =  np.full((self.ns, self.ni), -1, int)

        from_nodes_ctr = np.zeros(self.ns, int)
        for i in range(self.ni):
            ip_bit = int2bin(i, self.conv_k)[0]
            for j in range(self.ns):
                curr_st_bits = int2bin(j, self.constraint_length-1)
                if self.rsc:
                    fb_bit = self._binary_vecmul(curr_st_bits, self.fb_poly[1:])
                    new_bit = int2bin((ip_bit + fb_bit) % 2, 1)[0]
                else:
                    new_bit = ip_bit
                state_bits = [new_bit] + curr_st_bits 
                j_to = bin2int(state_bits[:-1])

                to_nodes[j][i] = j_to
                from_nodes[j_to][from_nodes_ctr[j_to]] = j

                op_bits = self._binary_matmul(state_bits)
                op_sym = bin2int(op_bits)
                op_mat[j, j_to] = op_sym
                op_by_tonode[j_to, from_nodes_ctr[j_to]] = op_sym
                ip_by_tonode[j_to, from_nodes_ctr[j_to]] = i
                op_by_fromnode[j][i] = op_sym
                from_nodes_ctr[j_to] += 1

        self.to_nodes = to_nodes
        self.from_nodes = from_nodes
        self.op_mat = op_mat
        self.ip_by_tonode = ip_by_tonode
        self.op_by_tonode = op_by_tonode
        self.op_by_fromnode = op_by_fromnode

class ConvEncoder:
    def __init__(self,
                 gen_poly=None,
                 rate=1/2,
                 constraint_length=3,
                 rsc=False,
                 terminate=False,
                 output_dtype=np.float32):

        if gen_poly is not None:
            if not all(isinstance(poly, str) for poly in gen_poly):
                raise ValueError("Each element of gen_poly must be a string.")
            if not all(len(poly) == len(gen_poly[0]) for poly in gen_poly):
                raise ValueError("Each polynomial must be of same length.")
            if not all(all(char in ['0','1'] for char in poly) for poly in gen_poly):
                raise ValueError("Each Polynomial must be a string of 0/1 s.")
            self._gen_poly = gen_poly
        else:
            valid_rates = (1/2, 1/3)
            valid_constraint_length = (3,4,5,6,7,8)
            if constraint_length not in valid_constraint_length:
                raise ValueError("Constraint length must be between 3 and 8.")
            if rate not in valid_rates:
                raise ValueError("Rate must be 1/3 or 1/2.")
            self._gen_poly = polynomial_selector(constraint_length)

        self._rsc = rsc
        self._terminate = terminate

        self._coderate_desired = 1/len(self._gen_poly)
        self._coderate = self._coderate_desired

        self._trellis = Trellis(self._gen_poly, rsc=self._rsc)
        self._mu = self.trellis._mu

        self._conv_k = self._trellis.conv_k
        self._conv_n = self._trellis.conv_n
        self._ni = 2 ** self._conv_k
        self._no = 2 ** self._conv_n
        self._ns = self._trellis.ns

        self._k = None
        self._n = None
        self.output_dtype = output_dtype

    @property
    def gen_poly(self):
        return self._gen_poly

    @property
    def coderate(self):
        if self._terminate and self._k is not None:
            term_factor = self._k / (self._k + self._mu)
            self._coderate = self._coderate_desired * term_factor
        return self._coderate

    @property
    def trellis(self):
        return self._trellis

    @property
    def terminate(self):
        return self._terminate

    @property
    def k(self):
        if self._k is None:
            print("Note: The value of k cannot be computed before the first call().")
        return self._k

    @property
    def n(self):
        if self._n is None:
            print("Note: The value of n cannot be computed before the first call().")
        return self._n

    def build(self, input_shape):
        # input_shape: (batch, k)
        self._k = input_shape[-1]
        self._n = int(self._k / self.coderate)
        # num_syms: 인코딩 주기 수 (conv_k 비트마다 한 주기)
        self.num_syms = self._k // self._conv_k

    def call(self, inputs):
        if inputs.ndim < 2:
            raise ValueError("Input must have rank > 1.")
        if self._k is None or inputs.shape[-1] != self._k:
            self.build(inputs.shape)
        msg = inputs.astype(np.int32)
        batch = msg.shape[0]

        output_shape = list(msg.shape)
        output_shape[-1] = self._n

        msg_reshaped = msg.reshape(batch, self._k)
        term_syms = self._mu if self._terminate else 0

        prev_st = np.zeros(batch, dtype=np.int32)
        encoded_list = []

        for idx in range(0, self._k, self._conv_k):
            msg_bits_idx = msg_reshaped[:, idx: idx+self._conv_k]
            msg_idx = bin2int_np(msg_bits_idx)  # shape: (batch,)
            new_st = self._trellis.to_nodes[prev_st, msg_idx]
            idx_syms = self._trellis.op_mat[prev_st, new_st]
            idx_bits = int2bin_np(idx_syms, self._conv_n)  # shape: (batch, conv_n)
            encoded_list.append(idx_bits)
            prev_st = new_st

        main_encoded = np.concatenate(encoded_list, axis=1)  # shape: (batch, num_syms * conv_n)

        if self._terminate:
            if self._rsc:
                fb_poly = np.array([int(x) for x in self.gen_poly[0][1:]], dtype=np.int32)
            term_list = []
            for idx in range(0, term_syms, self._conv_k):
                prev_st_bits = int2bin_np(prev_st, self._mu)  # shape: (batch, mu)
                if self._rsc:
                    msg_idx = np.mod(np.sum(fb_poly * prev_st_bits, axis=1), 2)
                else:
                    msg_idx = np.zeros(batch, dtype=np.int32)
                new_st = self._trellis.to_nodes[prev_st, msg_idx]
                idx_syms = self._trellis.op_mat[prev_st, new_st]
                idx_bits = int2bin_np(idx_syms, self._conv_n)
                term_list.append(idx_bits)
                prev_st = new_st
            term_encoded = np.concatenate(term_list, axis=1)
            cw = np.concatenate([main_encoded, term_encoded], axis=1)
        else:
            cw = main_encoded

        cw = cw.astype(self.output_dtype)
        cw_reshaped = cw.reshape(output_shape)
        return cw_reshaped

class RandomInterleaver:
    def __init__(self,
                 seed=None,
                 keep_batch_constant=True,
                 inverse=False,
                 keep_state=True,
                 axis=-1,
                 dtype=np.float32):

        self._keep_batch_constant = keep_batch_constant
        self._axis = axis
        self._inverse = inverse
        self._keep_state = keep_state
        self.dtype = dtype

        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        else:
            assert isinstance(seed, int), "seed must be int."

        self._seed_tuple = (1337, seed)

    @property
    def seed(self):
        return self._seed_tuple[1] 

    @property
    def axis(self):
        return self._axis

    @property
    def keep_state(self):
        return self._keep_state

    def _generate_perm_full(self, seq_length, batch_size, inverse=False):
        rand_seq = tf.random.stateless_uniform([batch_size, seq_length],
                                                self._seed_tuple,
                                                minval=0,
                                                maxval=1,
                                                dtype=tf.float32)

        perm_seq =  np.argsort(rand_seq.numpy(), axis=-1)

        if inverse:
            inv_perm_seq = np.empty_like(perm_seq)
            for i in range(batch_size):
                inv_perm_seq[i, perm_seq[i]] = np.arange(seq_length)
            perm_seq = inv_perm_seq

        return perm_seq

    def call_inverse(self, inputs, seed=None):
        input_shape = inputs.shape
        seq_length = input_shape[self._axis]
        batch_size = 1 if self._keep_batch_constant else input_shape[0]

        perm_seq = self._generate_perm_full(seq_length, batch_size, inverse=True)

        if self._keep_batch_constant:
            perm_seq = perm_seq[0]  # (seq_length,)
            x = np.take(inputs, perm_seq, axis=self._axis)
        else:
            x = np.empty_like(inputs)
            for i in range(input_shape[0]):
                x[i] = np.take(inputs[i], perm_seq[i], axis=self._axis)

        return x

    def call(self, inputs, seed=None):
        input_shape = inputs.shape
        seq_length = input_shape[self._axis]
        batch_size = 1 if self._keep_batch_constant else input_shape[0]

        perm_seq = self._generate_perm_full(seq_length, batch_size, inverse=False)

        if self._keep_batch_constant:
            perm_seq = perm_seq[0]  # (seq_length,)
            x = np.take(inputs, perm_seq, axis=self._axis)
        else:
            x = np.empty_like(inputs)
            for i in range(input_shape[0]):
                x[i] = np.take(inputs[i], perm_seq[i], axis=self._axis)

        return x

class Deinterleaver:
    def __init__(self, interleaver, dtype=np.float32):
        if not isinstance(interleaver, RandomInterleaver):
            raise ValueError("interleaver is not a valid interleaver instance.")
        self._interleaver = interleaver
        self.dtype = dtype

    @property
    def interleaver(self):
        return self._interleaver

    def call(self, inputs, seed=None):
        x = self._interleaver.call_inverse(inputs, seed=seed)
        return x.astype(self.dtype)

class TurboEncoder:
    def __init__(self, gen_poly=None, constraint_length=3, rate=1/3, terminate=False, 
                 interleaver_type='random', output_dtype=np.float32):
        if gen_poly is not None:
            self._gen_poly = gen_poly
        else:
            self._gen_poly = polynomial_selector(constraint_length)
        valid_rates = (1/2, 1/3)
        if rate not in valid_rates:
            raise ValueError("Invalid coderate.")
        self._coderate_desired = rate
        self._coderate = self._coderate_desired
        self._terminate = terminate
        self._interleaver_type = interleaver_type
        self.output_dtype = output_dtype
        rsc = True
        self._coderate_conv = 1 / len(self._gen_poly)  
        self._punct_pattern = puncture_pattern(rate, self._coderate_conv)

        self._trellis = Trellis(self._gen_poly, rsc=rsc)
        self._mu = self._trellis._mu
        self._conv_k = self._trellis.conv_k
        self._conv_n = self._trellis.conv_n
        self._ni = 2 ** self._conv_k
        self._no = 2 ** self._conv_n
        self._ns = self._trellis.ns
        self._k = None
        self._n = None
        if self._terminate:
            self.turbo_term = TurboTermination(self._mu + 1, conv_n=self._conv_n)
        self.internal_interleaver = RandomInterleaver(
                                                keep_batch_constant=True,
                                                seed=None,
                                                keep_state=True,
                                                axis=-1)
        self.convencoder = ConvEncoder(gen_poly=self._gen_poly, rsc=rsc, terminate=self._terminate)

        if self.punct_pattern is not None:
            self.punct_idx = np.argwhere(self.punct_pattern)

        if self._terminate:
            self.num_term_bits_ = int(
                self.turbo_term.get_num_term_syms()/self._coderate_conv)
            self.num_term_bits_punct = int(
                self.num_term_bits_*self._coderate_conv/self._coderate_desired)
        else:
            self.num_term_bits_ = 0
            self.num_term_bits_punct = 0

    @property
    def gen_poly(self):
        return self._gen_poly

    @property
    def coderate(self):
        if self._terminate and self._k is not None:
            term_factor = 1 + math.ceil(4 * self._mu / 3) / self._k
            self._coderate = self._coderate_desired / term_factor
        return self._coderate

    @property
    def trellis(self):
        return self._trellis

    @property
    def punct_pattern(self):
        return self._punct_pattern

    @property
    def k(self):
        if self._k is None:
            print("Note: The value of k cannot be computed before the first " \
                  "call().")
        return self._k

    @property
    def n(self):
        if self._n is None:
            print("Note: The value of n cannot be computed before the first " \
                  "call().")
        return self._n

    @property
    def terminate(self):
        return self._terminate

    def _puncture_cw(self, cw):
        cw = np.transpose(cw, (1, 2, 0))  # (n, 3, batch)
        cw_n = cw.shape[0]

        punct_period = self.punct_pattern.shape[0]
        mask_reps = cw_n // punct_period

        idx = np.tile(self.punct_idx, (mask_reps, 1))

        idx_per_period = self.punct_idx.shape[0]
        idx_per_time = idx_per_period / punct_period

        delta_times = cw_n - (mask_reps * punct_period)
        delta_idx_rows = int(delta_times * idx_per_time)

        time_offset = punct_period * np.arange(mask_reps)[None, :]
        row_idx = np.tile(time_offset, (idx_per_period, 1)).T.reshape(-1, 1)

        total_indices = mask_reps*idx_per_period + delta_idx_rows
        col_idx = np.zeros((total_indices,1), dtype=np.int32)

        if delta_times > 0:
            idx = np.concatenate([idx, self.punct_idx[:delta_idx_rows]], axis=0)
            time_n = punct_period * mask_reps
            row_idx_delta = np.tile(np.arange(time_n, time_n+delta_times)[None, :],
                                    (delta_idx_rows, 1))
            row_idx = np.concatenate([row_idx, row_idx_delta], axis=0)

        idx_offset = np.concatenate([row_idx, col_idx], axis=1).astype(np.int64)
        idx = idx + idx_offset  # shape (total_indices, 2)

        gathered = cw[idx[:, 0], idx[:, 1], :]  # (total_indices, batch)
        gathered = gathered.T  # (batch, total_indices)
        return gathered

    def build(self, input_shape):
        self._k = input_shape[-1]
        self._n = int(self._k / self._coderate_desired)  # e.g., k=10 => n=30
        self.num_syms = self._k // self._conv_k

    def encode(self, inputs):
        msg = inputs.astype(np.int32)
        batch = msg.shape[0]
        self.build(msg.shape)  # sets _k, _n, num_syms
        cw1_full = self.convencoder.call(msg)  # shape: (batch, n1)
        msg2 = self.internal_interleaver.call(msg)  # interleaved version of msg
        cw2_full = self.convencoder.call(msg2)  # shape: (batch, n1)
        preterm_n = int(self._k / self._coderate_conv)  # e.g., if _k=10 and coderate_conv=0.5 then preterm_n=20
        cw1_main = cw1_full[:, :preterm_n]  # shape (batch, preterm_n)
        cw2_main = cw2_full[:, :preterm_n]  # shape (batch, preterm_n)

        if self._terminate:
            term1 = cw1_full[:, preterm_n:]
            term2 = cw2_full[:, preterm_n:]
        par_idx = np.arange(1, preterm_n, self._conv_n)  # 예: for preterm_n=20 and conv_n=2 → indices: [1,3,...,19] (length = _k)
        cw2_par = cw2_main[:, par_idx]  # shape (batch, _k)
        cw1_main = cw1_main.reshape(batch, self._k, self._conv_n)
        cw2_par = cw2_par.reshape(batch, self._k, 1)
        cw_main = np.concatenate([cw1_main, cw2_par], axis=-1)

        if self._terminate:
            term_bits = self.turbo_term.termbits_conv2turbo(term1, term2)
            term_syms_turbo = term_bits.reshape([-1, self.num_term_bits_//2, 3])
            cw = np.concatenate([cw_main, term_syms_turbo], axis=-2)
        else:
            cw = cw_main

        if self.punct_pattern is not None:
            cw = self._puncture_cw(cw)

        return cw.astype(self.output_dtype)

class BCJRDecoder:
    def __init__(self, encoder=None, gen_poly=None, rate=1/2, constraint_length=3,
                 rsc=False, terminate=False, hard_out=True, algorithm='maxlog',
                 output_dtype=np.float32):
        if encoder is not None:
            self._gen_poly = encoder.gen_poly
            self._trellis = encoder.trellis
            self._terminate = encoder.terminate
        else:
            if gen_poly is not None:
                self._gen_poly = gen_poly
            else:
                self._gen_poly = polynomial_selector(constraint_length)
            self._trellis = Trellis(self._gen_poly, rsc=rsc)
            self._terminate = terminate

        self._conv_k = self._trellis.conv_k
        self._conv_n = self._trellis.conv_n
        self._mu = self._trellis._mu
        self._ni = 2 ** self._conv_k
        self._no = 2 ** self._conv_n
        self._ns = self._trellis.ns
        self._hard_out = hard_out
        self._algorithm = algorithm
        self._output_dtype = output_dtype
        self._k = None
        self._n = None
        self._num_syms = None
        self._coderate_desired = 1/len(self._gen_poly)
        self.ipst_op_idx, self.ipst_ip_idx = self._mask_by_tonode()

    def build(self, input_shape):
        # input_shape: (batch, n)
        if isinstance(input_shape, (list)):
            self._n = input_shape[-1]
        else:
            self._n = input_shape[0][-1]
        self._num_syms = int(self._n * self._coderate_desired)
        self._num_term_syms = self._mu if self._terminate else 0
        self._num_term_bits = int(self._num_term_syms / self._coderate_desired)
        self._k = self._num_syms - self._num_term_syms

    def _initialize(self, llr_ch):
        batch = llr_ch.shape[0]
        alpha = np.full((batch, self._ns), -np.inf, dtype=np.float32)
        alpha[:, 0] = 0.0
        if not self._terminate:
            beta = np.log(np.full((batch, self._ns), 1.0/self._ns, dtype=np.float32))
        else:
            beta = np.full((batch, self._ns), -np.inf, dtype=np.float32)
            beta[:, 0] = 0.0
        return alpha, beta

    def _mask_by_tonode(self):
        cnst = self._ns * self._ni
        from_nodes_vec = np.reshape(self._trellis.from_nodes, (cnst,))
        op_idx = np.reshape(self._trellis.op_by_tonode, (cnst,))
        st_op_idx = np.stack([from_nodes_vec, op_idx], axis=0)
        st_op_idx = np.transpose(st_op_idx, (1, 0))
        st_op_idx = st_op_idx.reshape((self._ns, self._ni, 2))

        ip_idx = np.reshape(self._trellis.ip_by_tonode, (cnst,))
        st_ip_idx = np.stack([from_nodes_vec, ip_idx], axis=0)
        st_ip_idx = np.transpose(st_ip_idx, (1, 0))
        st_ip_idx = st_ip_idx.reshape((self._ns, self._ni, 2))
        return st_op_idx, st_ip_idx

    def _bmcalc(self, llr_in):
        return bmcalc_numba(llr_in, self._conv_n, self._num_syms, self._no)

    def _update_fwd(self, alph_init, bm_mat, llr):
        return update_fwd_numba(alph_init, bm_mat, llr, self.ipst_ip_idx, 
                                  self._trellis.op_by_fromnode, self._num_syms)

    def _update_bwd(self, beta_init, bm_mat, llr, alph_list):
        return update_bwd_numba(beta_init, bm_mat, llr, alph_list, 
                                  self._trellis.op_by_fromnode, self._trellis.to_nodes, self._num_syms)

    def call(self, inputs):
        if isinstance(inputs, (tuple, list)):
            llr_ch, llr_apr = inputs
        else:
            llr_ch = inputs
            llr_apr = None

        output_shape = list(llr_ch.shape)
        if output_shape[-1] != self._n:
            if isinstance(inputs, (tuple, list)):
                self.build((inputs[0].shape, inputs[1].shape))
            else:
                self.build((llr_ch.shape,))
        output_shape[0] = -1
        output_shape[-1] = self._k
        llr_ch = np.reshape(llr_ch, (-1, self._n))
        if llr_apr is None:
            llr_apr = np.zeros((llr_ch.shape[0], self._num_syms), dtype=np.float32)

        llr_ch = -1. * llr_ch
        llr_apr = -1. * llr_apr

        bm_mat = self._bmcalc(llr_ch)
        alpha_init, beta_init = self._initialize(llr_ch)

        alph_ta = self._update_fwd(alpha_init, bm_mat, llr_apr)
        llr_op = self._update_bwd(beta_init, bm_mat, llr_apr, alph_ta)
        msghat = -1. * llr_op[..., :self._k]
        if self._hard_out:
            msghat = (msghat > 0)
        msghat = msghat.astype(self._output_dtype)
        msghat_reshaped = np.reshape(msghat, output_shape)
        return msghat_reshaped

class TurboDecoder:
    def __init__(self, encoder=None, gen_poly=None, rate=1/3, constraint_length=3,
                 interleaver='3GPP', terminate=False, num_iter=6, hard_out=True,
                 algorithm='maxlog', output_dtype=np.float32, llr_max=20.0):
        if encoder is not None:
            self._coderate = encoder._coderate
            self._gen_poly = encoder._gen_poly
            self._terminate = encoder._terminate
            self._trellis = encoder.trellis
            self.rsc = True
            self.internal_interleaver = encoder.internal_interleaver
        else:
            if gen_poly is not None:
                self._gen_poly = gen_poly
            else:
                self._gen_poly = polynomial_selector(constraint_length)
            self._coderate = rate
            self._terminate = terminate
            self.internal_interleaver = RandomInterleaver(
                seed=123,
                keep_batch_constant=True,
                keep_state=True,
                axis=-1,
                dtype=np.float32
            )
            self.rsc = True
            self._trellis = Trellis(self._gen_poly, rsc=self.rsc)
        self._coderate_conv = 1 / len(self._gen_poly)
        self._coderate_desired = rate
        self._mu = len(self._gen_poly[0]) - 1
        self._conv_k = self._trellis.conv_k
        self._conv_n = self._trellis.conv_n
        self._ni = 2 ** self._conv_k
        self._no = 2 ** self._conv_n
        self._ns = self._trellis.ns
        self._hard_out = hard_out
        self._algorithm = algorithm
        self._output_dtype = output_dtype
        self.num_iter = num_iter
        self.llr_max = llr_max

        self.bcjrdecoder = BCJRDecoder(encoder=encoder, gen_poly=self._gen_poly, rsc=True,
                                       terminate=self._terminate,
                                       hard_out=False, algorithm=algorithm,
                                       output_dtype=output_dtype)

        self.punct_pattern = puncture_pattern(self._coderate, self._coderate_conv)
        self._k = None
        self._n = None

        if self._terminate:
            self.turbo_term = TurboTermination(self._mu + 1, conv_n=self._conv_n)
            self._num_term_bits = 3 * self.turbo_term.get_num_term_syms()
        else:
            self._num_term_bits = 0

    def depuncture(self, y):
        y_depunct = np.zeros((self._depunct_len, y.shape[0]), dtype=y.dtype)
        indices = self._punct_indices.flatten() 
        y_depunct[indices, :] = y.T
        return y_depunct.T

    def _convenc_cws(self, y_turbo):
        y_turbo = self.depuncture(y_turbo)
        prepunct_n = int(self._n * 3 * self._coderate)
        msg_idx = np.arange(0, prepunct_n - self._num_term_bits)
        term_idx = np.arange(prepunct_n - self._num_term_bits, prepunct_n)
        y_cw = np.take(y_turbo, msg_idx, axis=-1)
        y_term = np.take(y_turbo, term_idx, axis=-1)
        enc1_sys_idx = np.arange(0, self._k * 3, 3).reshape(-1, 1)
        enc1_cw_idx = np.concatenate([enc1_sys_idx, enc1_sys_idx + 1], axis=1)
        enc1_cw_idx = np.squeeze(np.reshape(enc1_cw_idx, (-1, 2 * self._k)))
        y1_cw = np.take(y_cw, enc1_cw_idx, axis=-1)
        y1_sys_cw = np.take(y_cw, enc1_sys_idx, axis=-1)
        y1_sys_cw_squeezed = np.squeeze(y1_sys_cw, axis=-1)
        y2_sys_cw = self.internal_interleaver.call(y1_sys_cw_squeezed)[..., None]
        y2_nonsys_cw = np.take(y_cw, enc1_sys_idx + 2, axis=-1)
        y2_cw = np.stack([y2_sys_cw, y2_nonsys_cw], axis=-2)
        y2_cw = np.squeeze(y2_cw)
        y2_cw = np.reshape(y2_cw, (-1, 2 * self._k))
        if self._terminate:
            term_vec1, term_vec2 = self.turbo_term.term_bits_turbo2conv(y_term)
            y1_cw = np.concatenate([y1_cw, term_vec1], axis=1)
            y2_cw = np.concatenate([y2_cw, term_vec2], axis=1)
        return y1_cw, y2_cw

    def build(self, input_shape):
        if len(input_shape) < 2:
            raise ValueError("Input shape must have at least 2 dimensions")
        self._n = input_shape[-1]
        codefactor = self._coderate * 3
        turbo_n = int(self._n * codefactor)
        turbo_n_preterm = turbo_n - self._num_term_bits
        if turbo_n_preterm % 3 != 0:
            raise ValueError("Invalid codeword length for a terminated Turbo code")
        self._k = turbo_n_preterm // 3
        self._convenc_numsyms = self._k
        if self._terminate:
            self._convenc_numsyms += self._mu

        rate_factor = 3.0 * self._coderate
        self._depunct_len = int(rate_factor * self._n)
        punct_size = np.prod(self.punct_pattern.shape)
        rep_times = int(self._depunct_len // punct_size)

        mask_ = np.tile(self.punct_pattern, (rep_times, 1))
        extra_bits = int(self._depunct_len - rep_times * punct_size)

        if extra_bits > 0:
            extra_periods = int(extra_bits / 3)
            mask_ = np.concatenate([mask_, self.punct_pattern[:extra_periods, :]], axis=0)

        mask_ = np.squeeze(mask_.reshape(-1))
        self._punct_indices = np.argwhere(mask_).astype(np.int32)

    def decode(self, inputs):
        llr_max = self.llr_max
        if inputs.dtype != np.float32:
            raise ValueError("input must be np.float32")
        output_shape = list(inputs.shape)
        if output_shape[-1] != self._n:
            self.build(tuple(output_shape))
        llr_ch = np.reshape(inputs, (-1, self._n))
        output_shape[0] = -1
        output_shape[-1] = self._k

        y1_cw, y2_cw = self._convenc_cws(llr_ch)
        sys_idx = np.expand_dims(np.arange(0, self._k * 2, 2), axis=1)
        llr_ch = np.take(y1_cw, sys_idx, axis=-1)
        llr_ch = np.squeeze(llr_ch, axis=-1)
        llr_ch2 = np.take(y2_cw, sys_idx, axis=-1)
        llr_ch2 = np.squeeze(llr_ch2, axis=-1)
        llr_1e = np.zeros((llr_ch.shape[0], self._convenc_numsyms), dtype=np.float32)
        term_info_bits = self._mu if self._terminate else 0
        llr_terminfo = np.zeros((llr_ch.shape[0], term_info_bits), dtype=np.float32)
        L_2 = np.zeros_like(llr_ch2)

        for _ in range(self.num_iter):
            L_1 = self.bcjrdecoder.call((y1_cw, llr_1e))
            L_1 = L_1[..., :self._k]
            llr_extr = L_1 - llr_ch - llr_1e[..., :self._k]
            llr_2e = self.internal_interleaver.call(llr_extr)
            llr_2e = np.concatenate([llr_2e, llr_terminfo], axis=-1)
            llr_2e = np.clip(llr_2e, -llr_max, llr_max)
            L_2 = self.bcjrdecoder.call((y2_cw, llr_2e))
            L_2 = L_2[..., :self._k]
            llr_extr = L_2 - llr_2e[..., :self._k] - llr_ch2
            llr_1e = self.internal_interleaver.call_inverse(llr_extr)
            llr_1e = np.clip(llr_1e, -llr_max, llr_max)
            llr_1e = np.concatenate([llr_1e, llr_terminfo], axis=-1)

        output = self.internal_interleaver.call_inverse(L_2)

        if self._hard_out:
            output = (0.0 < output).astype(np.int32)
        else:
            output = output.astype(self._output_dtype)

        output = np.reshape(output, output_shape)
        return output

class TurboTermination:
    def __init__(self, constraint_length, conv_n=2, num_conv_encs=2, num_bitstreams=3):
        self.mu_ = constraint_length - 1
        self.conv_n = conv_n
        if num_conv_encs != 2:
            raise ValueError("num_conv_encs must be 2")
        self.num_conv_encs = num_conv_encs
        self.num_bitstreams = num_bitstreams

    def get_num_term_syms(self):
        total_term_bits = self.conv_n * self.num_conv_encs * self.mu_
        turbo_term_syms = math.ceil(total_term_bits / self.num_bitstreams)
        return turbo_term_syms

    def termbits_conv2turbo(self, term_bits1, term_bits2):
        term_bits = np.concatenate([term_bits1, term_bits2], axis=-1)
        num_term_bits = term_bits.shape[-1]
        num_term_syms = math.ceil(num_term_bits / self.num_bitstreams)
        extra_bits = self.num_bitstreams * num_term_syms - num_term_bits
        if extra_bits > 0:
            zeros = np.zeros((term_bits.shape[0], extra_bits), dtype=np.float32)
            term_bits = np.concatenate([term_bits, zeros], axis=-1)
        return term_bits

    def term_bits_turbo2conv(self, term_bits):
        input_len = term_bits.shape[-1]
        if input_len % self.num_bitstreams != 0:
            raise ValueError("Programming Error: input not divisible by num_bitstreams")
        enc1_term_idx = np.arange(0, self.conv_n * self.mu_)
        enc2_term_idx = np.arange(self.conv_n * self.mu_, 2 * self.conv_n * self.mu_)
        term_bits1 = term_bits[:, enc1_term_idx]
        term_bits2 = term_bits[:, enc2_term_idx]
        return term_bits1, term_bits2

def polynomial_selector(constraint_length):
    gen_poly_dict = {
        3: ('111', '101'),  # (7, 5)
        4: ('1011', '1101'),  # (13, 15)
        5: ('10011', '11011'),  # (23, 33)
        6: ('111101', '101011'),  # (75, 53)
    }
    return gen_poly_dict[constraint_length]

def bin2int(arr):
    if len(arr) == 0: return None
    return int(''.join([str(x) for x in arr]), 2)

def int2bin(num, len_):
    assert num >= 0,  "Input integer should be non-negative"
    assert len_ >= 0,  "width should be non-negative"

    bin_ = format(num, f'0{len_}b')
    binary_vals = [int(x) for x in bin_[-len_:]] if len_ else []
    return binary_vals

def bin2int_np(arr):
    arr = np.asarray(arr, dtype=np.int32)
    n = arr.shape[-1]
    shifts = np.arange(n-1, -1, -1, dtype=np.int32)
    return np.sum(np.left_shift(arr, shifts), axis=-1)

def int2bin_np(ints, length):
    assert length >= 0, "length should be non-negative"
    ints = np.asarray(ints, dtype=np.int32)
    shifts = np.arange(length-1, -1, -1, dtype=np.int32)
    bits = np.mod(np.right_shift(np.expand_dims(ints, -1), shifts), 2)
    return bits.astype(np.int32)

def puncture_pattern(turbo_coderate, conv_coderate):
    if turbo_coderate == 1/2:
        pattern = [[1, 1, 0], [1, 0, 1]]
    elif turbo_coderate == 1/3:
        pattern = [[1, 1, 1]]

    turbo_punct_pattern = np.asarray(pattern, dtype=np.bool_)  # bool 타입 변환
    return turbo_punct_pattern
