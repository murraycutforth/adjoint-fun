import unittest
import logging
import tempfile

from PIL import Image
import numpy as np
import fenics as fe

from src.image_utils.image_utils import image_to_array, numpy_array_to_fenics_fn


logging.getLogger('FFC').setLevel(logging.WARNING)



class TestImageProcessing(unittest.TestCase):
    def test_image_to_grayscale(self):
        # Create a grayscale image
        img = Image.fromarray(np.array([[0, 127], [255, 127]], dtype='uint8'), 'L')

        with tempfile.NamedTemporaryFile(suffix=".png") as fp:

            img.save(fp.name)

            result = image_to_array(fp.name)

        np.testing.assert_array_equal(result, np.array([[0, 127], [255, 127]], dtype='uint8'))


    def test_numpy_array_to_fenics_fn(self):
        mesh = fe.UnitSquareMesh(32, 32)
        V = fe.FunctionSpace(mesh, 'P', 1)

        result = numpy_array_to_fenics_fn(np.array([[0, 0, 255, 255], [0, 0, 255, 255], [255, 255, 0, 0], [255, 255, 0, 0]], dtype='uint8'), V)

        self.assertTrue(isinstance(result, fe.function.function.Function))
        self.assertAlmostEqual(result(0.125, 0.125), 1.0, places=2)  # should be close to black (0)
        self.assertAlmostEqual(result(0.875, 0.875), 1.0, places=2)  # should be close to white (1)


if __name__ == '__main__':
    unittest.main()

