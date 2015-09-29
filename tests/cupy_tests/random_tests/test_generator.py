import mock
import operator
import os
import unittest

import numpy
import six

import cupy
from cupy import cuda
from cupy.cuda import curand
from cupy.random import generator
from cupy import testing


@testing.gpu
class TestRandomState(unittest.TestCase):

    _multiprocess_can_split_ = True
    args = (0.0, 1.0)
    size = None

    def setUp(self):
        self.rs = generator.RandomState()

    def check_lognormal(self, curand_func, dtype):
        shape = cupy._get_size(self.size)
        exp_size = six.moves.reduce(operator.mul, shape, 1)
        if exp_size % 2 == 1:
            exp_size += 1

        curand_func.return_value = cupy.zeros(exp_size, dtype=dtype)
        out = self.rs.lognormal(self.args[0], self.args[1], self.size, dtype)
        gen, _, size, mean, sigma = curand_func.call_args[0]
        self.assertIs(gen, self.rs._generator)
        self.assertEqual(size, exp_size)
        self.assertIs(mean, self.args[0])
        self.assertIs(sigma, self.args[1])
        self.assertEqual(out.shape, shape)

    def test_lognormal_float(self):
        curand.generateLogNormalDouble = mock.Mock()
        self.check_lognormal(curand.generateLogNormalDouble, float)

    def test_lognormal_float32(self):
        curand.generateLogNormal = mock.Mock()
        self.check_lognormal(curand.generateLogNormal, numpy.float32)

    def test_lognormal_float64(self):
        curand.generateLogNormalDouble = mock.Mock()
        self.check_lognormal(curand.generateLogNormalDouble, numpy.float64)

    def check_normal(self, curand_func, dtype):
        shape = cupy._get_size(self.size)
        exp_size = six.moves.reduce(operator.mul, shape, 1)
        if exp_size % 2 == 1:
            exp_size += 1

        curand_func.return_value = cupy.zeros(exp_size, dtype=dtype)
        out = self.rs.normal(self.args[0], self.args[1], self.size, dtype)
        gen, _, size, loc, scale = curand_func.call_args[0]
        self.assertIs(gen, self.rs._generator)
        self.assertEqual(size, exp_size)
        self.assertIs(loc, self.args[0])
        self.assertIs(scale, self.args[1])
        self.assertEqual(out.shape, shape)

    def test_normal_float32(self):
        curand.generateNormal = mock.Mock()
        self.check_normal(curand.generateNormal, numpy.float32)

    def test_normal_float64(self):
        curand.generateNormalDouble = mock.Mock()
        self.check_normal(curand.generateNormalDouble, numpy.float64)

    def check_random_sample(self, curand_func, dtype):
        out = self.rs.random_sample(self.size, dtype)
        curand_func.assert_called_once_with(
            self.rs._generator, out.data.ptr, out.size)

    def test_random_sample_float32(self):
        curand.generateUniform = mock.Mock()
        self.check_random_sample(curand.generateUniform, numpy.float32)

    def test_random_sample_float64(self):
        curand.generateUniformDouble = mock.Mock()
        self.check_random_sample(curand.generateUniformDouble, numpy.float64)

    def check_seed(self, curand_func, seed):
        self.rs.seed(seed)
        call_args_list = curand_func.call_args_list
        self.assertEqual(1, len(call_args_list))
        call_args = call_args_list[0][0]
        self.assertEqual(2, len(call_args))
        self.assertIs(self.rs._generator, call_args[0])
        self.assertEqual(numpy.uint64, call_args[1].dtype)

    def test_seed_none(self):
        curand.setPseudoRandomGeneratorSeed = mock.Mock()
        self.check_seed(curand.setPseudoRandomGeneratorSeed, None)

    @testing.for_all_dtypes()
    def test_seed_not_none(self, dtype):
        curand.setPseudoRandomGeneratorSeed = mock.Mock()
        self.check_seed(curand.setPseudoRandomGeneratorSeed, dtype(0))


@testing.gpu
class TestRandomState2(TestRandomState):

    args = (10.0, 20.0)
    size = None


@testing.gpu
class TestRandomState3(TestRandomState):

    args = (0.0, 1.0)
    size = 10


@testing.gpu
class TestRandomState4(TestRandomState):

    args = (0.0, 1.0)
    size = (1, 2, 3)


@testing.gpu
class TestRandomState6(TestRandomState):

    args = (0.0, 1.0)
    size = 3


@testing.gpu
class TestRandomState7(TestRandomState):

    args = (0.0, 1.0)
    size = (3, 3)


@testing.gpu
class TestRandomState8(TestRandomState):

    args = (0.0, 1.0)
    size = ()


@testing.gpu
class TestRandAndRandN(unittest.TestCase):

    def setUp(self):
        self.rs = generator.RandomState()

    def test_rand(self):
        self.rs.random_sample = mock.Mock()
        self.rs.rand(1, 2, 3, dtype=numpy.float32)
        self.rs.random_sample.assert_called_once_with(
            size=(1, 2, 3), dtype=numpy.float32)

    def test_rand_invalid_argument(self):
        with self.assertRaises(TypeError):
            self.rs.rand(1, 2, 3, unnecessary='unnecessary_argument')

    def test_randn(self):
        self.rs.normal = mock.Mock()
        self.rs.randn(1, 2, 3, dtype=numpy.float32)
        self.rs.normal.assert_called_once_with(
            size=(1, 2, 3), dtype=numpy.float32)

    def test_randn_invalid_argument(self):
        with self.assertRaises(TypeError):
            self.rs.randn(1, 2, 3, unnecessary='unnecessary_argument')


class TestResetStates(unittest.TestCase):

    def test_reset_states(self):
        generator._random_states = 'dummy'
        generator.reset_states()
        self.assertEqual({}, generator._random_states)


@testing.gpu
class TestGetRandomState(unittest.TestCase):

    def setUp(self):
        self.device_id = cuda.Device().id
        self.rs_tmp = generator._random_states

    def tearDown(self, *args):
        generator._random_states = self.rs_tmp

    def test_get_random_state_initialize(self):
        generator._random_states = {}
        rs = generator.get_random_state()
        self.assertEqual(generator._random_states[self.device_id], rs)

    def test_get_random_state_memoized(self):
        generator._random_states = {self.device_id: 'expected',
                                    self.device_id + 1: 'dummy'}
        rs = generator.get_random_state()
        self.assertEqual('expected', generator._random_states[self.device_id])
        self.assertEqual('dummy', generator._random_states[self.device_id + 1])
        self.assertEqual('expected', rs)


@testing.gpu
class TestGetRandomState2(unittest.TestCase):

    def setUp(self):
        self.rs_tmp = generator.RandomState
        generator.RandomState = mock.Mock()
        self.rs_dict = generator._random_states
        generator._random_states = {}

    def tearDown(self, *args):
        generator.RandomState = self.rs_tmp
        generator._random_states = self.rs_dict

    def test_get_random_state_no_chainer_seed(self):
        os.unsetenv('CHAINER_SEED')
        generator.get_random_state()
        generator.RandomState.assert_called_with(None)

    def test_get_random_state_with_chainer_seed(self):
        os.environ['CHAINER_SEED'] = '1'
        generator.get_random_state()
        generator.RandomState.assert_called_with('1')


class TestCheckAndGetDtype(unittest.TestCase):

    @testing.for_float_dtypes(no_float16=True)
    def test_float32_64_type(self, dtype):
        self.assertEqual(generator._check_and_get_dtype(dtype),
                         numpy.dtype(dtype))

    def test_float16(self):
        with self.assertRaises(TypeError):
            generator._check_and_get_dtype(numpy.float16)

    @testing.for_int_dtypes()
    def test_int_type(self, dtype):
        with self.assertRaises(TypeError):
            generator._check_and_get_dtype(dtype)