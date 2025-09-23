import tensorflow as tf
import numpy as np
import scipy as sp
from importlib_resources import files, as_file
import codes # pylint: disable=relative-beyond-top-level
import numbers # to check if n, k are numbers
from abc import ABC
from abc import abstractmethod
import random
import types

dtypes = {
    'single' : {
        'tf' : {
            'cdtype' : tf.complex64,
            'rdtype' : tf.float32
        },
        'np' : {
            'cdtype' : np.complex64,
            'rdtype' : np.float32
        }
    },
    'double' : {
        'tf' : {
            'cdtype' : tf.complex128,
            'rdtype' : tf.float64
        },
        'np' : {
            'cdtype' : np.complex128,
            'rdtype' : np.float64
        }
    }
}

class Config():
    # This object is a singleton
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            instance = object.__new__(cls)
            cls._instance = instance
        return cls._instance

    def __init__(self):
        self._seed = None
        self._py_rng = None
        self._np_rng = None
        self._tf_rng = None
        self._precision = None

        self.precision = 'single'

    @property
    def py_rng(self):
        if self._py_rng is None:
            self._py_rng = random.Random()
        return self._py_rng

    @property
    def np_rng(self):
        if self._np_rng is None:
            self._np_rng = np.random.default_rng()
        return self._np_rng

    @property
    def tf_rng(self):
        if self._tf_rng  is None:
            self._tf_rng = tf.random.Generator.from_non_deterministic_state()
        return self._tf_rng

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        if seed is not None:
            seed = int(seed)
        self._seed = seed

        self.tf_rng.reset_from_seed(seed)

        self.py_rng.seed(seed)

        self._np_rng = np.random.default_rng(seed)

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, v):
        if v not in ["single", "double"]:
            raise ValueError("Precision must be ``single`` or ``double``.")
        self._precision = v

    @property
    def np_rdtype(self):
        """
        `np.dtype` : Default NumPy dtype for real floating point numbers
        """
        return dtypes[self.precision]['np']['rdtype']

    @property
    def np_cdtype(self):
        """
        `np.dtype` : Default NumPy dtype for complex floating point numbers
        """
        return dtypes[self.precision]['np']['cdtype']

    @property
    def tf_rdtype(self):
        """
        `tf.dtype` : Default TensorFlow dtype for real floating point numbers
        """
        return dtypes[self.precision]['tf']['rdtype']

    @property
    def tf_cdtype(self):
        """
        `tf.dtype` : Default TensorFlow dtype for complex floating point numbers
        """
        return dtypes[self.precision]['tf']['cdtype']

config = Config()

class Object(ABC):
    """Abstract class for Sionna PHY objects

    Parameters
    ----------
    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations.
        If set to `None`, the default
        :attr:`~sionna.phy.config.Config.precision` is used.
    """

    def __init__(self, *args, precision=None, **kwargs):
        if precision is None:
            self._precision = config.precision
        elif precision in ['single', 'double']:
            self._precision = precision
        else:
            raise ValueError("'precision' must be 'single' or 'double'")

    @property
    def precision(self):
        """
        `str`, "single" | "double" : Precision used for all compuations
        """
        return self._precision

    @property
    def cdtype(self):
        """
        `tf.complex` : Type for complex floating point numbers
        """
        return dtypes[self.precision]['tf']['cdtype']

    @property
    def rdtype(self):
        """
        `tf.float` : Type for real floating point numbers
        """
        return dtypes[self.precision]['tf']['rdtype']

    def _cast_or_check_precision(self, v):
        """Cast tensor to internal precision or check
           if a variable has the right precision
        """
        # Check correct dtype for Variables
        if isinstance(v, tf.Variable):
            if v.dtype.is_complex:
                if v.dtype != self.cdtype:
                    msg = f"Wrong dtype. Expected {self.cdtype}" + \
                          f", got {v.dtype}"
                    raise ValueError(msg)
            elif v.dtype.is_floating:
                if v.dtype != self.rdtype:
                    msg = f"Wrong dtype. Expected {self.cdtype}" + \
                          f", got {v.dtype}"
                    raise ValueError("Wrong dtype")

        # Cast tensors to the correct dtype
        else:
            if not isinstance(v, tf.Tensor):
                v = tf.convert_to_tensor(v)
            if v.dtype.is_complex:
                v = tf.cast(v, self.cdtype)
            else:
                v = tf.cast(v, self.rdtype)

        return v

class Block(Object):
    def __init__(self, *args, precision=None, **kwargs):
        super().__init__(precision=precision, **kwargs)
        self._built = False

    @property
    def built(self):
        return self._built

    def build(self, *arg_shapes, **kwarg_shapes):
        pass

    @abstractmethod
    def call(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    def _convert_to_tensor(self, v):
        """Casts floating or complex tensors to the block's precision"""
        if isinstance(v, np.ndarray):
            v = tf.convert_to_tensor(v)
        if isinstance(v, tf.Tensor):
            if v.dtype.is_floating:
                v = tf.cast(v, self.rdtype)
            elif v.dtype.is_complex:
                v = tf.cast(v, self.cdtype)
        return v

    def _get_shape(self, v):
        """Converts an input to the corresponding TensorShape"""
        try :
            v = tf.convert_to_tensor(v)
        except (TypeError, ValueError):
            pass
        if hasattr(v, "shape"):
            return tf.TensorShape(v.shape)
        else:
            return tf.TensorShape([])

    def __call__(self, *args, **kwargs):

        args, kwargs = tf.nest.map_structure(self._convert_to_tensor,
                                             [args, kwargs])
        with tf.init_scope(): # pylint: disable=not-context-manager
            if not self._built:
                shapes =  tf.nest.map_structure(self._get_shape,
                                             [args, kwargs])
                self.build(*shapes[0], **shapes[1])
                self._built = True

        return self.call(*args, **kwargs)

class LDPC5GEncoder(Block):
    def __init__(self,
                 k,
                 n,
                 num_bits_per_symbol=None,
                 bg=None,
                 precision=None,
                 **kwargs):

        super().__init__(precision=precision, **kwargs)

        if not isinstance(k, numbers.Number):
            raise TypeError("k must be a number.")
        if not isinstance(n, numbers.Number):
            raise TypeError("n must be a number.")
        k = int(k) # k or n can be float (e.g. as result of n=k*r)
        n = int(n) # k or n can be float (e.g. as result of n=k*r)

        if k>8448:
            raise ValueError("Unsupported code length (k too large).")
        if k<12:
            raise ValueError("Unsupported code length (k too small).")

        if n>(316*384):
            raise ValueError("Unsupported code length (n too large).")
        if n<0:
            raise ValueError("Unsupported code length (n negative).")

        # init encoder parameters
        self._k = k # number of input bits (= input shape)
        self._n = n # the desired length (= output shape)
        self._coderate = k / n
        self._check_input = True # check input for consistency (i.e., binary)

        # allow actual code rates slightly larger than 948/1024
        # to account for the quantization procedure in 38.214 5.1.3.1
        if self._coderate>(948/1024): # as specified in 38.212 5.4.2.1
            print(f"Warning: effective coderate r>948/1024 for n={n}, k={k}.")
        if self._coderate>(0.95): # as specified in 38.212 5.4.2.1
            raise ValueError(f"Unsupported coderate (r>0.95) for n={n}, k={k}.")
        if self._coderate<(1/5):
            # outer rep. coding currently not supported
            raise ValueError("Unsupported coderate (r<1/5).")

        # construct the basegraph according to 38.212
        # if bg is explicitly provided
        self._bg = self._sel_basegraph(self._k, self._coderate, bg)

        self._z, self._i_ls, self._k_b = self._sel_lifting(self._k, self._bg)
        self._bm = self._load_basegraph(self._i_ls, self._bg)

        # total number of codeword bits
        self._n_ldpc = self._bm.shape[1] * self._z
        # if K_real < K _target puncturing must be applied earlier
        self._k_ldpc = self._k_b * self._z

        # construct explicit graph via lifting
        pcm = self._lift_basegraph(self._bm, self._z)

        pcm_a, pcm_b_inv, pcm_c1, pcm_c2 = self._gen_submat(self._bm,
                                                            self._k_b,
                                                            self._z,
                                                            self._bg)

        # init sub-matrices for fast encoding ("RU"-method)
        # note: dtype is tf.float32;
        self._pcm = pcm # store the sparse parity-check matrix (for decoding)

        # store indices for fast gathering (instead of explicit matmul)
        self._pcm_a_ind = self._mat_to_ind(pcm_a)
        self._pcm_b_inv_ind = self._mat_to_ind(pcm_b_inv)
        self._pcm_c1_ind = self._mat_to_ind(pcm_c1)
        self._pcm_c2_ind = self._mat_to_ind(pcm_c2)

        self._num_bits_per_symbol = num_bits_per_symbol
        if num_bits_per_symbol is not None:
            self._out_int, self._out_int_inv  = self.generate_out_int(self._n,
                                                    self._num_bits_per_symbol)

    ###############################
    # Public methods and properties
    ###############################

    @property
    def k(self):
        """Number of input information bits"""
        return self._k

    @property
    def n(self):
        "Number of output codeword bits"
        return self._n

    @property
    def coderate(self):
        """Coderate of the LDPC code after rate-matching"""
        return self._coderate

    @property
    def k_ldpc(self):
        """Number of LDPC information bits after rate-matching"""
        return self._k_ldpc

    @property
    def n_ldpc(self):
        """Number of LDPC codeword bits before rate-matching"""
        return self._n_ldpc

    @property
    def pcm(self):
        """Parity-check matrix for given code parameters"""
        return self._pcm

    @property
    def z(self):
        """Lifting factor of the basegraph"""
        return self._z

    @property
    def num_bits_per_symbol(self):
        """Modulation order used for the rate-matching output interleaver"""
        return self._num_bits_per_symbol

    @property
    def out_int(self):
        """Output interleaver sequence as defined in 5.4.2.2"""
        return self._out_int
    @property
    def out_int_inv(self):
        """Inverse output interleaver sequence as defined in 5.4.2.2"""
        return self._out_int_inv

    #################
    # Utility methods
    #################

    def generate_out_int(self, n, num_bits_per_symbol):
        # allow float inputs, but verify that they represent integer
        if n%1!=0:
            raise ValueError("n must be int.")
        if num_bits_per_symbol%1!=0:
            raise ValueError("num_bits_per_symbol must be int.")
        n = int(n)
        if n<=0:
            raise ValueError("n must be a positive integer.")
        if num_bits_per_symbol<=0:
            raise ValueError("num_bits_per_symbol must be a positive integer.")
        num_bits_per_symbol = int(num_bits_per_symbol)

        if n%num_bits_per_symbol!=0:
            raise ValueError("n must be a multiple of num_bits_per_symbol.")

        # pattern as defined in Sec 5.4.2.2
        perm_seq = np.zeros(n, dtype=int)
        for j in range(int(n/num_bits_per_symbol)):
            for i in range(num_bits_per_symbol):
                perm_seq[i + j*num_bits_per_symbol] \
                    = int(i * int(n/num_bits_per_symbol) + j)

        perm_seq_inv = np.argsort(perm_seq)

        return perm_seq, perm_seq_inv

    def _sel_basegraph(self, k, r, bg_=None):
        """Select basegraph according to [3GPPTS38212_LDPC]_ and check for consistency."""

        # if bg is explicitly provided, we only check for consistency
        if bg_ is None:
            if k <= 292:
                bg = "bg2"
            elif k <= 3824 and r <= 0.67:
                bg = "bg2"
            elif r <= 0.25:
                bg = "bg2"
            else:
                bg = "bg1"
        elif bg_ in ("bg1", "bg2"):
            bg = bg_
        else:
            raise ValueError("Basegraph must be bg1, bg2 or None.")

        # check for consistency
        if bg=="bg1" and k>8448:
            raise ValueError("K is not supported by BG1 (too large).")

        if bg=="bg2" and k>3840:
            raise ValueError(
                f"K is not supported by BG2 (too large) k ={k}.")

        if bg=="bg1" and r<1/3:
            raise ValueError("Only coderate>1/3 supported for BG1. \
            Remark: Repetition coding is currently not supported.")

        if bg=="bg2" and r<1/5:
            raise ValueError("Only coderate>1/5 supported for BG2. \
            Remark: Repetition coding is currently not supported.")

        return bg

    def _load_basegraph(self, i_ls, bg):
        """Helper to load basegraph from csv files.

        ``i_ls`` is sub_index of the basegraph and fixed during lifting
        selection.
        """

        if i_ls > 7:
            raise ValueError("i_ls too large.")

        if i_ls < 0:
            raise ValueError("i_ls cannot be negative.")

        # csv files are taken from 38.212 and dimension is explicitly given
        if bg=="bg1":
            bm = np.zeros([46, 68]) - 1 # init matrix with -1 (None positions)
        elif bg=="bg2":
            bm = np.zeros([42, 52]) - 1 # init matrix with -1 (None positions)
        else:
            raise ValueError("Basegraph not supported.")

        # and load the basegraph from csv format in folder "codes"
        source = files(codes).joinpath(f"5G_{bg}.csv")
        with as_file(source) as codes.csv:
            bg_csv = np.genfromtxt(codes.csv, delimiter=";")

        # reconstruct BG for given i_ls
        r_ind = 0
        for r in np.arange(2, bg_csv.shape[0]):
            # check for next row index
            if not np.isnan(bg_csv[r, 0]):
                r_ind = int(bg_csv[r, 0])
            c_ind = int(bg_csv[r, 1]) # second column in csv is column index
            value = bg_csv[r, i_ls + 2] # i_ls entries start at offset 2
            bm[r_ind, c_ind] = value

        return bm

    def _lift_basegraph(self, bm, z):
        """Lift basegraph with lifting factor ``z`` and shifted identities as
        defined by the entries of ``bm``."""

        num_nonzero = np.sum(bm>=0) # num of non-neg elements in bm

        # init all non-zero row/column indices
        r_idx = np.zeros(z*num_nonzero)
        c_idx = np.zeros(z*num_nonzero)
        data = np.ones(z*num_nonzero)

        # row/column indices of identity matrix for lifting
        im = np.arange(z)

        idx = 0
        for r in range(bm.shape[0]):
            for c in range(bm.shape[1]):
                if bm[r,c]==-1: # -1 is used as all-zero matrix placeholder
                    pass #do nothing (sparse)
                else:
                    # roll matrix by bm[r,c]
                    c_roll = np.mod(im+bm[r,c], z)
                    # append rolled identity matrix to pcm
                    r_idx[idx*z:(idx+1)*z] = r*z + im
                    c_idx[idx*z:(idx+1)*z] = c*z + c_roll
                    idx += 1

        # generate lifted sparse matrix from indices
        pcm = sp.sparse.csr_matrix((data,(r_idx, c_idx)),
                                   shape=(z*bm.shape[0], z*bm.shape[1]))
        return pcm

    def _sel_lifting(self, k, bg):
        # lifting set according to 38.212 Tab 5.3.2-1
        s_val = [[2, 4, 8, 16, 32, 64, 128, 256],
                [3, 6, 12, 24, 48, 96, 192, 384],
                [5, 10, 20, 40, 80, 160, 320],
                [7, 14, 28, 56, 112, 224],
                [9, 18, 36, 72, 144, 288],
                [11, 22, 44, 88, 176, 352],
                [13, 26, 52, 104, 208],
                [15, 30, 60, 120, 240]]

        if bg == "bg1":
            k_b = 22
        else:
            if k > 640:
                k_b = 10
            elif k > 560:
                k_b = 9
            elif k > 192:
                k_b = 8
            else:
                k_b = 6

        # find the min of Z from Tab. 5.3.2-1 s.t. k_b*Z>=K'
        min_val = 100000
        z = 0
        i_ls = 0
        i = -1
        for s in s_val:
            i += 1
            for s1 in s:
                x = k_b *s1
                if  x >= k:
                    # valid solution
                    if x < min_val:
                        min_val = x
                        z = s1
                        i_ls = i

        # and set K=22*Z for bg1 and K=10Z for bg2
        if bg == "bg1":
            k_b = 22
        else:
            k_b = 10

        return z, i_ls, k_b

    def _gen_submat(self, bm, k_b, z, bg):
        """Split the basegraph into multiple sub-matrices such that efficient
        encoding is possible.
        """
        g = 4 # code property (always fixed for 5G)
        mb = bm.shape[0] # number of CN rows in basegraph (BG property)

        bm_a = bm[0:g, 0:k_b]
        bm_b = bm[0:g, k_b:(k_b+g)]
        bm_c1 = bm[g:mb, 0:k_b]
        bm_c2 = bm[g:mb, k_b:(k_b+g)]

        # H could be sliced immediately (but easier to implement if based on B)
        hm_a = self._lift_basegraph(bm_a, z)

        # not required for encoding, but helpful for debugging
        # hm_b = self._lift_basegraph(bm_b, z)

        hm_c1 = self._lift_basegraph(bm_c1, z)
        hm_c2 = self._lift_basegraph(bm_c2, z)

        hm_b_inv = self._find_hm_b_inv(bm_b, z, bg)

        return hm_a, hm_b_inv, hm_c1, hm_c2

    def _find_hm_b_inv(self, bm_b, z, bg):
        # permutation indices
        pm_a= int(bm_b[0,0])
        if bg=="bg1":
            pm_b_inv = int(-bm_b[1, 0])
        else: # structure of B is slightly different for bg2
            pm_b_inv = int(-bm_b[2, 0])

        hm_b_inv = np.zeros([4*z, 4*z])

        im = np.eye(z)

        am = np.roll(im, pm_a, axis=1)
        b_inv = np.roll(im, pm_b_inv, axis=1)
        ab_inv = np.matmul(am, b_inv)

        # row 0
        hm_b_inv[0:z, 0:z] = b_inv
        hm_b_inv[0:z, z:2*z] = b_inv
        hm_b_inv[0:z, 2*z:3*z] = b_inv
        hm_b_inv[0:z, 3*z:4*z] = b_inv

        # row 1
        hm_b_inv[z:2*z, 0:z] = im + ab_inv
        hm_b_inv[z:2*z, z:2*z] = ab_inv
        hm_b_inv[z:2*z, 2*z:3*z] = ab_inv
        hm_b_inv[z:2*z, 3*z:4*z] = ab_inv

        # row 2
        if bg=="bg1":
            hm_b_inv[2*z:3*z, 0:z] = ab_inv
            hm_b_inv[2*z:3*z, z:2*z] = ab_inv
            hm_b_inv[2*z:3*z, 2*z:3*z] = im + ab_inv
            hm_b_inv[2*z:3*z, 3*z:4*z] = im + ab_inv
        else: # for bg2 the structure is slightly different
            hm_b_inv[2*z:3*z, 0:z] = im + ab_inv
            hm_b_inv[2*z:3*z, z:2*z] = im + ab_inv
            hm_b_inv[2*z:3*z, 2*z:3*z] = ab_inv
            hm_b_inv[2*z:3*z, 3*z:4*z] = ab_inv

        # row 3
        hm_b_inv[3*z:4*z, 0:z] = ab_inv
        hm_b_inv[3*z:4*z, z:2*z] = ab_inv
        hm_b_inv[3*z:4*z, 2*z:3*z] = ab_inv
        hm_b_inv[3*z:4*z, 3*z:4*z] = im + ab_inv

        # return results as sparse matrix
        return sp.sparse.csr_matrix(hm_b_inv)

    def _mat_to_ind(self, mat):
        """Helper to transform matrix into index representation for
        tf.gather. An index pointing to the `last_ind+1` is used for non-existing edges due to irregular degrees."""
        m = mat.shape[0]
        n = mat.shape[1]

        # transpose mat for sorted column format
        c_idx, r_idx, _ = sp.sparse.find(mat.transpose())

        # sort indices explicitly, as scipy.sparse.find changed from column to
        # row sorting in scipy>=1.11
        idx = np.argsort(r_idx)
        c_idx = c_idx[idx]
        r_idx = r_idx[idx]

        # find max number of no-zero entries
        n_max = np.max(mat.getnnz(axis=1))

        # init index array with n (pointer to last_ind+1, will be a default
        # value)
        gat_idx = np.zeros([m, n_max]) + n

        r_val = -1
        c_val = 0
        for idx in range(len(c_idx)):
            # check if same row or if a new row starts
            if r_idx[idx] != r_val:
                r_val = r_idx[idx]
                c_val = 0
            gat_idx[r_val, c_val] = c_idx[idx]
            c_val += 1

        gat_idx = tf.cast(tf.constant(gat_idx), tf.int32)
        return gat_idx

    def _matmul_gather(self, mat, vec):
        """Implements a fast sparse matmul via gather function."""

        # add 0 entry for gather-reduce_sum operation
        # (otherwise ragged Tensors are required)
        bs = tf.shape(vec)[0]
        vec = tf.concat([vec, tf.zeros([bs, 1], dtype=self.rdtype)], 1)

        retval = tf.gather(vec, mat, batch_dims=0, axis=1)
        retval = tf.reduce_sum(retval, axis=-1)

        return retval

    def _encode_fast(self, s):
        """Main encoding function based on gathering function."""
        p_a = self._matmul_gather(self._pcm_a_ind, s)
        p_a = self._matmul_gather(self._pcm_b_inv_ind, p_a)

        # calc second part of parity bits p_b
        # second parities are given by C_1*s' + C_2*p_a' + p_b' = 0
        p_b_1 = self._matmul_gather(self._pcm_c1_ind, s)
        p_b_2 = self._matmul_gather(self._pcm_c2_ind, p_a)
        p_b = p_b_1 + p_b_2

        c = tf.concat([s, p_a, p_b], 1)

        # faster implementation of mod-2 operation c = tf.math.mod(c, 2)
        c_uint8 = tf.cast(c, tf.uint8)
        c_bin = tf.bitwise.bitwise_and(c_uint8, tf.constant(1, tf.uint8))
        c = tf.cast(c_bin, self.rdtype)

        c = tf.expand_dims(c, axis=-1) # returns nx1 vector
        return c

    def build(self, input_shape):
        """"Build block."""
        # check if k and input shape match
        if input_shape[-1]!=self._k:
            raise ValueError("Last dimension must be of length k.")

    def encode(self, bits):
        bits = tf.cast(bits, tf.float32)
        return self.call(bits)

    def call(self, bits):
        # Reshape inputs to [...,k]
        input_shape = bits.get_shape().as_list()
        new_shape = [-1, input_shape[-1]]
        u = tf.reshape(bits, new_shape)

        # assert if bits are non-binary
        if self._check_input:
            tf.debugging.assert_equal(
                tf.reduce_min(
                    tf.cast(
                        tf.logical_or(
                            tf.equal(u, tf.constant(0, self.rdtype)),
                            tf.equal(u, tf.constant(1, self.rdtype)),
                            ),
                        self.rdtype)),
                tf.constant(1, self.rdtype),
                "Input must be binary.")
            # input datatype consistency should be only evaluated once
            self._check_input = False

        batch_size = tf.shape(u)[0]

        # add "filler" bits to last positions to match info bit length k_ldpc
        u_fill = tf.concat([u,
            tf.zeros([batch_size, self._k_ldpc-self._k], self.rdtype)],axis=1)

        # use optimized encoding based on tf.gather
        c = self._encode_fast(u_fill)

        c = tf.reshape(c, [batch_size, self._n_ldpc]) # remove last dim

        # remove filler bits at pos (k, k_ldpc)
        c_no_filler1 = tf.slice(c, [0, 0], [batch_size, self._k])
        c_no_filler2 = tf.slice(c,
                               [0, self._k_ldpc],
                               [batch_size, self._n_ldpc - self._k_ldpc])

        c_no_filler = tf.concat([c_no_filler1, c_no_filler2], 1)

        # shorten the first 2*Z positions and end after n bits
        # (remaining parity bits can be used for HARQ)
        c_short = tf.slice(c_no_filler, [0, 2*self._z], [batch_size, self.n])

        if self._num_bits_per_symbol is not None:
            c_short = tf.gather(c_short, self._out_int, axis=-1)

        # Reshape c_short so that it matches the original input dimensions
        output_shape = input_shape[0:-1] + [self.n]
        output_shape[0] = -1
        c_reshaped = tf.reshape(c_short, output_shape)

        return c_reshaped
    
class LDPCBPDecoder(Block):
    def __init__(self,
                 pcm,
                 cn_update="boxplus-phi",
                 vn_update="sum",
                 cn_schedule="flooding",
                 hard_out=True,
                 num_iter=20,
                 llr_max=20.,
                 v2c_callbacks=None,
                 c2v_callbacks=None,
                 return_state=False,
                 precision=None,
                 **kwargs):

        super().__init__(precision=precision, **kwargs)

        # check inputs for consistency
        if not isinstance(hard_out, bool):
            raise TypeError('hard_out must be bool.')
        if not isinstance(num_iter, int):
            raise TypeError('num_iter must be int.')
        if num_iter<0:
            raise ValueError('num_iter cannot be negative.')
        if not isinstance(return_state, bool):
            raise TypeError('return_state must be bool.')

        if isinstance(pcm, np.ndarray):
            if not np.array_equal(pcm, pcm.astype(bool)):
                raise ValueError('PC matrix must be binary.')
        elif isinstance(pcm, sp.sparse.csr_matrix):
            if not np.array_equal(pcm.data, pcm.data.astype(bool)):
                raise ValueError('PC matrix must be binary.')
        elif isinstance(pcm, sp.sparse.csc_matrix):
            if not np.array_equal(pcm.data, pcm.data.astype(bool)):
                raise ValueError('PC matrix must be binary.')
        else:
            raise TypeError("Unsupported dtype of pcm.")

        # Deprecation warning for cn_type
        if 'cn_type' in kwargs:
            raise TypeError("'cn_type' is deprecated; use 'cn_update' instead.")

        # init decoder parameters
        self._pcm = pcm
        self._hard_out = hard_out
        self._num_iter = tf.constant(num_iter, dtype=tf.int32)
        self._return_state = return_state

        self._num_cns = pcm.shape[0] # total number of check nodes
        self._num_vns = pcm.shape[1] # total number of variable nodes

        # internal value for llr clipping
        if not isinstance(llr_max, (int, float)):
            raise TypeError("llr_max must be int or float.")
        self._llr_max = tf.cast(llr_max, self.rdtype)

        if v2c_callbacks is None:
            self._v2c_callbacks = []
        else:
            if isinstance(v2c_callbacks, (list, tuple)):
                self._v2c_callbacks = v2c_callbacks
            elif isinstance(v2c_callbacks, types.FunctionType):
                # allow that user provides single function
                self._v2c_callbacks = [v2c_callbacks,]
            else:
                raise TypeError("v2c_callbacks must be a list of callables.")

        if c2v_callbacks is None:
            self._c2v_callbacks = []
        else:
            if isinstance(c2v_callbacks, (list, tuple)):
                self._c2v_callbacks = c2v_callbacks
            elif isinstance(c2v_callbacks, types.FunctionType):
                # allow that user provides single function
                self._c2v_callbacks = [c2v_callbacks,]
            else:
                raise TypeError("c2v_callbacks must be a list of callables.")

        if isinstance(cn_schedule, str) and cn_schedule=="flooding":
            self._scheduling = "flooding"
            self._cn_schedule = tf.stack([tf.range(self._num_cns)], axis=0)
        elif tf.is_tensor(cn_schedule) or isinstance(cn_schedule, np.ndarray):
            cn_schedule = tf.cast(cn_schedule, tf.int32)
            self._scheduling = "custom"
            # check custom schedule for consistency
            if len(cn_schedule.shape)!=2:
                raise ValueError("cn_schedule must be of rank 2.")
            if tf.reduce_max(cn_schedule)>=self._num_cns:
                msg = "cn_schedule can only contain values smaller number_cns."
                raise ValueError(msg)
            if tf.reduce_min(cn_schedule)<0:
                msg = "cn_schedule cannot contain negative values."
                raise ValueError(msg)
            self._cn_schedule = cn_schedule
        else:
            msg = "cn_schedule can be 'flooding' or an array of ints."
            raise ValueError(msg)

        if isinstance(pcm, np.ndarray):
            pcm = sp.sparse.csr_matrix(pcm)

        self._cn_idx, self._vn_idx, _ = sp.sparse.find(pcm)

        idx = np.argsort(self._vn_idx)
        self._cn_idx = self._cn_idx[idx]
        self._vn_idx = self._vn_idx[idx]

        self._num_edges = len(self._vn_idx)

        # pre-load the CN function
        if cn_update=='boxplus':
            # check node update using the tanh function
            self._cn_update = cn_update_tanh
        elif cn_update=='boxplus-phi':
            # check node update using the "_phi" function
            self._cn_update = cn_update_phi
        elif cn_update in ('minsum', 'min'):
            # check node update using the min-sum approximation
            self._cn_update = cn_update_minsum
        elif cn_update=="offset-minsum":
            # check node update using the min-sum approximation
            self._cn_update = cn_update_offset_minsum
        elif cn_update=='identity':
            self._cn_update = cn_node_update_identity
        elif isinstance(cn_update, types.FunctionType):
            self._cn_update = cn_update
        else:
            raise TypeError("Provided cn_update not supported.")

        # pre-load the VN function
        if vn_update=='sum':
            self._vn_update = vn_update_sum
        elif vn_update=='identity':
            self._vn_update = vn_node_update_identity
        elif isinstance(vn_update, types.FunctionType):
            self._vn_update = vn_update
        else:
            raise TypeError("Provided vn_update not supported.")

        v2c_perm = np.argsort(self._cn_idx)
        # and the inverse operation;
        v2c_perm_inv = np.argsort(v2c_perm)
        # only required for layered decoding
        self._v2c_perm_inv = tf.constant(v2c_perm_inv)
        self._v2c_perm = tf.RaggedTensor.from_value_rowids(
                                values=v2c_perm,
                                value_rowids=self._cn_idx[v2c_perm])

        self._c2v_perm = tf.RaggedTensor.from_value_rowids(
                                values=v2c_perm_inv,
                                value_rowids=self._vn_idx)

    def decode(self, llr_ch, /, *, num_iter=None, msg_v2c=None):
        """Iterative BP decoding function.
        
        Args:
            llr_ch: Log-likelihood ratios from the channel.
            num_iter: Number of decoding iterations. If None, self.num_iter is used.
            msg_v2c: Optional initial variable-to-check messages for warm start decoding.
            
        Returns:
            Decoded bits (hard) or LLRs (soft) depending on the hard_out setting.
        """
        return self.call(llr_ch, num_iter=num_iter, msg_v2c=msg_v2c)

    def _bp_iter(self, msg_v2c, msg_c2v, llr_ch, x_hat, it, num_iter):
        # Unroll loop to keep XLA / Keras compatibility
        # For flooding this will be unrolled to a single loop iteration
        for j in range(self._cn_schedule.shape[0]):

            # get active check nodes
            if self._scheduling=="flooding":
                # for flooding all CNs are active
                v2c_perm = self._v2c_perm
            else: # select active CNs for j-th subiteration
                cn_idx = tf.gather(self._cn_schedule, j, axis=0)
                v2c_perm = tf.gather(self._v2c_perm, cn_idx, axis=0)

            # Gather ragged tensor of incoming messages at CN.
            # The shape is [num_cns, None, batch_size,...].
            # The None dimension is the ragged dimension and depends on the
            # individual check node degree

            msg_cn_rag = tf.gather(msg_v2c, v2c_perm, axis=0)

            # Apply the CN update
            msg_cn_rag_ = self._cn_update(msg_cn_rag, self._llr_max)

            # Apply CN callbacks
            for cb in self._c2v_callbacks:
                msg_cn_rag_ = cb(msg_cn_rag_, it)

            # Apply partial message updates for layered decoding
            if self._scheduling!="flooding":
                # note: the scatter update operation is quite expensive
                up_idx = tf.gather(self._c2v_perm.flat_values,
                                   v2c_perm.flat_values)
                # update only active cns are updated
                msg_c2v = tf.tensor_scatter_nd_update(
                                     msg_c2v,
                                     tf.expand_dims(up_idx, axis=1),
                                     msg_cn_rag_.flat_values)
            else:
                # for flodding all nodes are updated
                msg_c2v = msg_cn_rag_.flat_values

            msg_vn_rag = tf.gather(msg_c2v, self._c2v_perm, axis=0)

            # Apply the VN update
            msg_vn_rag_, x_hat = self._vn_update(msg_vn_rag,
                                                 llr_ch,
                                                 self._llr_max)

            # apply v2c callbacks
            for cb in self._v2c_callbacks:
                msg_vn_rag_ = cb(msg_vn_rag_, it+1, x_hat)

            # we return flat values to avoid ragged tensors passing the tf.
            # while boundary (possible issues with XLA)
            msg_v2c = msg_vn_rag_.flat_values

        #increase iteration coutner
        it += 1

        return msg_v2c, msg_c2v, llr_ch, x_hat, it, num_iter

    # pylint: disable=unused-argument,unused-variable
    def _stop_cond(self, msg_v2c, msg_c2v, llr_ch, x_hat, it, num_iter):
        return it < num_iter

    # pylint: disable=(unused-argument)
    def build(self, input_shape, **kwargs):
        # Raise AssertionError if shape of x is invalid

        assert (input_shape[-1]==self._num_vns), \
                            'Last dimension must be of length n.'

    def call(self, llr_ch, /, *, num_iter=None, msg_v2c=None):
        """Iterative BP decoding function.
        """

        if num_iter is None:
            num_iter=self._num_iter

        # clip LLRs for numerical stability
        llr_ch = tf.clip_by_value(llr_ch,
                                  clip_value_min=-self._llr_max,
                                  clip_value_max=self._llr_max)

        # reshape to support multi-dimensional inputs
        llr_ch_shape = llr_ch.get_shape().as_list()
        new_shape = [-1, self._num_vns]
        llr_ch_reshaped = tf.reshape(llr_ch, new_shape)

        # batch dimension is last dimension due to ragged tensor representation
        llr_ch = tf.transpose(llr_ch_reshaped, (1, 0))

        # logits are converted into "true" LLRs as usually done in literature
        llr_ch *= -1.

        if msg_v2c is None:
            # init v2c messages with channel LLRs
            msg_v2c = tf.gather(llr_ch, self._vn_idx)
        else:
            msg_v2c *= -1 # invert sign due to logit definition

        msg_c2v = tf.zeros_like(msg_v2c)

        # apply VN callbacks before first iteration
        if self._v2c_callbacks != []:
            msg_vn_rag_ = tf.RaggedTensor.from_value_rowids(
                                values=msg_v2c,
                                value_rowids=self._vn_idx)
            # apply v2c callbacks
            for cb in self._v2c_callbacks:
                msg_vn_rag_ = cb(msg_vn_rag_, tf.constant(0, tf.int32), llr_ch)

            # Ensure shape as otherwise XLA cannot infer
            # the output signature of the loop
            msg_v2c = msg_vn_rag_.flat_values

        inputs = (msg_v2c, msg_c2v, llr_ch, llr_ch,
                  tf.constant(0, tf.int32), num_iter)

        # and run main decoding loop for num_iter iterations
        msg_v2c, _, _, x_hat, _, _ = tf.while_loop(
                                        self._stop_cond,self._bp_iter,
                                        inputs, maximum_iterations=num_iter)

        ######################
        # Post process outputs
        ######################

        # restore batch dimension to first dimension
        x_hat = tf.transpose(x_hat, (1,0))

        if self._hard_out: # hard decide decoder output if required
            x_hat = tf.greater_equal(tf.cast(0, self.rdtype), x_hat)
            x_hat = tf.cast(x_hat, self.rdtype)
        else:
            x_hat *= -1.  # convert LLRs back into logits

        # Reshape c_short so that it matches the original input dimensions
        output_shape = llr_ch_shape
        output_shape[0] = -1 # Dynamic batch dim
        x_reshaped = tf.reshape(x_hat, output_shape)

        if not self._return_state:
            return x_reshaped
        else:
            msg_v2c *= -1 # invert sign due to logit definition
            return x_reshaped, msg_v2c

# pylint: disable=unused-argument,unused-variable
def vn_node_update_identity(msg_c2v_rag, llr_ch, llr_clipping=None, **kwargs):
    # aggregate all incoming messages per node
    x_tot = tf.reduce_sum(msg_c2v_rag, axis=1) + llr_ch

    return msg_c2v_rag, x_tot

def vn_update_sum(msg_c2v_rag, llr_ch, llr_clipping=None):
    # aggregate all incoming messages per node
    x = tf.reduce_sum(msg_c2v_rag, axis=1)
    x_tot = tf.add(x, llr_ch)

    x_e = tf.ragged.map_flat_values(lambda x,y,row_ind: x+tf.gather(y, row_ind),
                            -1.*msg_c2v_rag, x_tot, msg_c2v_rag.value_rowids())

    if llr_clipping is not None:
        x_e = tf.clip_by_value(x_e,
                    clip_value_min=-llr_clipping, clip_value_max=llr_clipping)
        x_tot = tf.clip_by_value(x_tot,
                    clip_value_min=-llr_clipping, clip_value_max=llr_clipping)
    return x_e, x_tot

# pylint: disable=unused-argument,unused-variable
def cn_node_update_identity(msg_v2c_rag, *kwargs):
    return msg_v2c_rag

def cn_update_offset_minsum(msg_v2c_rag, llr_clipping=None, offset=0.5):
    def _sign_val_minsum(msg):
        """Helper to replace find sign-value during min-sum decoding.
        Must be called with `map_flat_values`."""

        sign_val = tf.sign(msg)
        sign_val = tf.where(tf.equal(sign_val, 0),
                            tf.ones_like(sign_val),
                            sign_val)
        return sign_val

    # a constant used to overwrite the first min
    large_val = 100000.
    msg_v2c_rag = tf.clip_by_value(msg_v2c_rag,
                               clip_value_min=-large_val,
                               clip_value_max=large_val)

    # only output is clipped (we assume input was clipped by previous function)

    # calculate sign of outgoing msg and the node
    sign_val = tf.ragged.map_flat_values(_sign_val_minsum, msg_v2c_rag)
    sign_node = tf.reduce_prod(sign_val, axis=1)

    sign_val = tf.ragged.map_flat_values(
                                    lambda x, y, row_ind:
                                    tf.multiply(x, tf.gather(y, row_ind)),
                                    sign_val,
                                    sign_node,
                                    sign_val.value_rowids())

    # remove sign from messages
    msg = tf.ragged.map_flat_values(tf.abs, msg_v2c_rag)

    min_val = tf.reduce_min(msg, axis=1, keepdims=True)

    msg_min1 = tf.ragged.map_flat_values(lambda x, y, row_ind:
                                            x - tf.gather(y, row_ind),
                                            msg,
                                            tf.squeeze(min_val, axis=1),
                                            msg.value_rowids())

    msg = tf.ragged.map_flat_values(
                        lambda x: tf.where(tf.equal(x, 0), large_val, x),
                        msg_min1)

    min_val_2 = tf.reduce_min(msg, axis=1, keepdims=True) + min_val

    node_sum = tf.reduce_sum(msg, axis=1, keepdims=True) - (2*large_val-1.)
    # indicator that duplicated min was detected (per node)
    double_min = 0.5*(1-tf.sign(node_sum))

    min_val_e = (1-double_min) * min_val + (double_min) * min_val_2

    min_1 = tf.squeeze(tf.gather(min_val, msg.value_rowids()), axis=1)
    min_e = tf.squeeze(tf.gather(min_val_e, msg.value_rowids()), axis=1)
    msg_e = tf.ragged.map_flat_values(
                lambda x: tf.where(x==large_val, min_e, min_1), msg)

    msg_e = tf.ragged.map_flat_values(
                lambda x: tf.ensure_shape(x, msg.flat_values.shape), msg_e)

    msg_e = tf.ragged.map_flat_values(lambda x,y: tf.maximum(x-y, 0),
                                      msg_e, offset)

    msg = tf.ragged.map_flat_values(tf.multiply, sign_val, msg_e)

    if llr_clipping is not None:
        msg = tf.clip_by_value(msg,
                    clip_value_min=-llr_clipping, clip_value_max=llr_clipping)
    return msg

def cn_update_minsum(msg_v2c_rag, llr_clipping=None):
    msg_c2v = cn_update_offset_minsum(msg_v2c_rag,
                                      llr_clipping=llr_clipping,
                                      offset=0)
    return msg_c2v

def cn_update_tanh(msg, llr_clipping=None):
    atanh_clip_value = 1 - 1e-7

    msg = msg / 2
    msg = tf.ragged.map_flat_values(tf.tanh, msg) # tanh is not overloaded

    msg = tf.ragged.map_flat_values(
            lambda x: tf.where(tf.equal(x, 0), tf.ones_like(x) * 1e-12, x), msg)

    msg_prod = tf.reduce_prod(msg, axis=1)

    msg = tf.ragged.map_flat_values(
                        lambda x,y,row_ind : x * tf.gather(y, row_ind),
                        msg**-1, msg_prod, msg.value_rowids())

    msg = tf.ragged.map_flat_values(
        lambda x: tf.where(tf.less(tf.abs(x), 1e-7), tf.zeros_like(x), x), msg)

    msg = tf.clip_by_value(msg,
                           clip_value_min=-atanh_clip_value,
                           clip_value_max=atanh_clip_value)

    # atanh is not overloaded for ragged tensors
    msg = 2 * tf.ragged.map_flat_values(tf.atanh, msg)

    # clip output values if required
    if llr_clipping is not None:
        msg = tf.clip_by_value(msg,
                               clip_value_min=-llr_clipping,
                               clip_value_max=llr_clipping)
    return msg

def cn_update_phi(msg, llr_clipping=None):
    def _phi(x):
        """Improved phi function implementation for numerical stability.
        
        The phi function is its own inverse: phi(phi(x)) = x
        For numerical stability, use different formulas based on input magnitude.
        """
        if x.dtype==tf.float32:
            # Clip values for numerical stability
            x = tf.clip_by_value(x, 8.5e-8, 16.635532)
            
            # Two different implementations based on input size
            small_x = tf.less(x, 10.0)
            
            # Traditional formula for smaller values
            # phi(x) = log((exp(x) + 1) / (exp(x) - 1))
            exp_x = tf.exp(x)
            traditional_result = tf.math.log((exp_x + 1.0) / (exp_x - 1.0))
            
            # Approximation for larger values: phi(x) ≈ 2 * e^(-x)
            approx_result = 2.0 * tf.exp(-x)
            
            # Choose implementation based on input size
            return tf.where(small_x, traditional_result, approx_result)
            
        elif x.dtype==tf.float64:
            x = tf.clip_by_value(x, 1e-12, 28.324079)
            
            # Same approach for double precision
            small_x = tf.less(x, 10.0)
            exp_x = tf.exp(x)
            traditional_result = tf.math.log((exp_x + 1.0) / (exp_x - 1.0))
            approx_result = 2.0 * tf.exp(-x)
            
            return tf.where(small_x, traditional_result, approx_result)
        else:
            raise TypeError("Unsupported dtype for phi function.")

    sign_val = tf.sign(msg)
    sign_val = tf.ragged.map_flat_values(lambda x : tf.where(tf.equal(x, 0),
                                         tf.ones_like(x), x), sign_val)
    sign_node = tf.reduce_prod(sign_val, axis=1)

    sign_val = tf.ragged.map_flat_values(
                lambda x,y,row_ind : x * tf.gather(y, row_ind),
                sign_val, sign_node, sign_val.value_rowids())

    msg = tf.ragged.map_flat_values(tf.abs, msg) # remove sign
    msg = tf.ragged.map_flat_values(_phi, msg)
    msg_sum = tf.reduce_sum(msg, axis=1)

    msg = tf.ragged.map_flat_values(
                            lambda x, y, row_ind : x + tf.gather(y, row_ind),
                            -1.*msg, msg_sum, msg.value_rowids())

    sign_val = sign_val.with_flat_values(tf.stop_gradient(sign_val.flat_values))
    msg_e = sign_val * tf.ragged.map_flat_values(_phi, msg)

    if llr_clipping is not None:
        msg_e = tf.clip_by_value(msg_e,
                    clip_value_min=-llr_clipping, clip_value_max=llr_clipping)
    return msg_e

class LDPC5GDecoder(LDPCBPDecoder):
    def __init__(self,
                 encoder,
                 cn_update="boxplus-phi",
                 vn_update="sum",
                 cn_schedule="flooding",
                 hard_out=True,
                 return_infobits=True,
                 num_iter=20,
                 llr_max=20.,
                 v2c_callbacks=None,
                 c2v_callbacks=None,
                 prune_pcm=True,
                 return_state=False,
                 precision=None,
                 **kwargs):

        # needs the 5G Encoder to access all 5G parameters
        if not isinstance(encoder, LDPC5GEncoder):
            raise TypeError("encoder must be of class LDPC5GEncoder.")

        self._encoder = encoder
        pcm = encoder.pcm

        if not isinstance(return_infobits, bool):
            raise TypeError('return_info must be bool.')
        self._return_infobits = return_infobits

        if not isinstance(return_state, bool):
            raise TypeError('return_state must be bool.')
        self._return_state = return_state

        # Deprecation warning for cn_type
        if 'cn_type' in kwargs:
            raise TypeError("'cn_type' is deprecated; use 'cn_update' instead.")

        if not isinstance(prune_pcm, bool):
            raise TypeError('prune_pcm must be bool.')
        self._prune_pcm = prune_pcm
        if prune_pcm:
            # find index of first position with only degree-1 VN
            dv = np.sum(pcm, axis=0) # VN degree
            last_pos = encoder._n_ldpc
            for idx in range(encoder._n_ldpc-1, 0, -1):
                if dv[0, idx]==1:
                    last_pos = idx
                else:
                    break
            # number of filler bits
            k_filler = self.encoder.k_ldpc - self.encoder.k

            # number of punctured bits
            nb_punc_bits = ((self.encoder.n_ldpc - k_filler)
                                     - self.encoder.n - 2*self.encoder.z)

            # if layered decoding is used, qunatized number of punctured bits
            # to a multiple of z; otherwise scheduling groups of Z CNs becomes
            # impossible
            if cn_schedule=="layered":
                nb_punc_bits = np.floor(nb_punc_bits/self.encoder.z) \
                             * self.encoder.z
                nb_punc_bits = int (nb_punc_bits) # cast to int

            # effective codeword length after pruning of vn-1 nodes
            self._n_pruned = np.max((last_pos, encoder._n_ldpc - nb_punc_bits))
            self._nb_pruned_nodes = encoder._n_ldpc - self._n_pruned

            # remove last CNs and VNs from pcm
            pcm = pcm[:-self._nb_pruned_nodes, :-self._nb_pruned_nodes]

            #check for consistency
            if self._nb_pruned_nodes<0:
                msg = "Internal error: number of pruned nodes must be positive."
                raise ArithmeticError(msg)
        else:
            # no pruning; same length as before
            self._nb_pruned_nodes = 0
            self._n_pruned = encoder._n_ldpc

        if cn_schedule=="layered":
            z = self._encoder.z
            num_blocks = int(pcm.shape[0]/z)
            cn_schedule = []
            for i in range(num_blocks):
                cn_schedule.append(np.arange(z) + i*z)
            cn_schedule = tf.stack(cn_schedule, axis=0)

        super().__init__(pcm,
                         cn_update=cn_update,
                         vn_update=vn_update,
                         cn_schedule=cn_schedule,
                         hard_out=hard_out,
                         num_iter=num_iter,
                         llr_max=llr_max,
                         v2c_callbacks=v2c_callbacks,
                         c2v_callbacks=c2v_callbacks,
                         return_state=return_state,
                         precision=precision,
                         **kwargs)

    @property
    def encoder(self):
        """LDPC Encoder used for rate-matching/recovery"""
        return self._encoder


    def build(self, input_shape, **kwargs):
        """Build block"""

        if input_shape[-1]!=self.encoder.n:
            raise ValueError('Last dimension must be of length n.')

        self._old_shape_5g = input_shape

    def decode(self, llr_ch, /, *, num_iter=None, msg_v2c=None):
        """Iterative BP decoding function and rate matching.
        
        Args:
            llr_ch: Log-likelihood ratios from the channel.
            num_iter: Number of decoding iterations. If None, self.num_iter is used.
            msg_v2c: Optional initial variable-to-check messages for warm start decoding.
            
        Returns:
            Decoded bits (hard) or LLRs (soft) depending on the hard_out setting.
            Only information bits are returned if return_infobits=True.
        """
        return self.call(llr_ch, num_iter=num_iter, msg_v2c=msg_v2c).numpy()

    def call(self, llr_ch, /, *, num_iter=None, msg_v2c=None):
        """Iterative BP decoding function and rate matching.
        """
        if isinstance(llr_ch, np.ndarray):
            llr_ch = tf.convert_to_tensor(llr_ch, dtype=self.rdtype)

        llr_ch_shape = llr_ch.get_shape().as_list()
        new_shape = [-1, self.encoder.n]
        llr_ch_reshaped = tf.reshape(llr_ch, new_shape)
        batch_size = tf.shape(llr_ch_reshaped)[0]

        if self._encoder.num_bits_per_symbol is not None:
            llr_ch_reshaped = tf.gather(llr_ch_reshaped,
                                        self._encoder.out_int_inv,
                                        axis=-1)

        llr_5g = tf.concat(
                    [tf.zeros([batch_size, 2*self.encoder.z], self.rdtype),
                    llr_ch_reshaped], axis=1)

        k_filler = self.encoder.k_ldpc - self.encoder.k # number of filler bits
        nb_punc_bits = ((self.encoder.n_ldpc - k_filler)
                        - self.encoder.n - 2*self.encoder.z)

        llr_5g = tf.concat([llr_5g,
                    tf.zeros([batch_size, nb_punc_bits - self._nb_pruned_nodes],
                            self.rdtype)], axis=1)

        x1 = tf.slice(llr_5g, [0,0], [batch_size, self.encoder.k])

        nb_par_bits = (self.encoder.n_ldpc - k_filler
                       - self.encoder.k - self._nb_pruned_nodes)
        x2 = tf.slice(llr_5g,
                      [0, self.encoder.k],
                      [batch_size, nb_par_bits])

        z = -tf.cast(self._llr_max, self.rdtype) \
            * tf.ones([batch_size, k_filler], self.rdtype)

        llr_5g = tf.concat([x1, z, x2], axis=1)

        output = super().call(llr_5g, num_iter=num_iter, msg_v2c=msg_v2c)

        if self._return_state:
            x_hat, msg_v2c = output
        else:
            x_hat = output

        if self._return_infobits:# return only info bits
            u_hat = tf.slice(x_hat, [0,0], [batch_size, self.encoder.k])
            output_shape = llr_ch_shape[0:-1] + [self.encoder.k]
            output_shape[0] = -1
            u_reshaped = tf.reshape(u_hat, output_shape)

            if self._return_state:
                return u_reshaped, msg_v2c
            else:
                return u_reshaped

        else: # return all codeword bits
            x = tf.reshape(x_hat, [batch_size, self._n_pruned])

            x_no_filler1 = tf.slice(x, [0, 0], [batch_size, self.encoder.k])

            x_no_filler2 = tf.slice(x,
                                    [0, self.encoder.k_ldpc],
                                    [batch_size,
                                    self._n_pruned-self.encoder.k_ldpc])

            x_no_filler = tf.concat([x_no_filler1, x_no_filler2], 1)

            x_short = tf.slice(x_no_filler,
                               [0, 2*self.encoder.z],
                               [batch_size, self.encoder.n])

            if self._encoder.num_bits_per_symbol is not None:
                x_short = tf.gather(x_short, self._encoder.out_int, axis=-1)

            llr_ch_shape[0] = -1
            x_short= tf.reshape(x_short, llr_ch_shape)

            if self._return_state:
                return x_short, msg_v2c
            else:
                return x_short
