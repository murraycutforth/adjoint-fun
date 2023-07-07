from PIL import Image
import numpy as np
import fenics as fe
import numpy as np
from scipy.interpolate import RectBivariateSpline



def image_to_array(image_path: str) -> np.array:

    # Open the image file.
    image = Image.open(image_path)

    # Convert the image to grayscale.
    image = image.convert("L")

    # Convert the image data to a numpy array.
    numpy_array = np.array(image)

    # Transform the numpy array (images usually start from top left, but FEniCS starts from bottom left)
    numpy_array = np.flipud(numpy_array)
    numpy_array = numpy_array.T

    return numpy_array


def numpy_array_to_fenics_fn(numpy_array: np.array, V: fe.FunctionSpace, arr_height: float = 1.0, arr_width: float = 1.0):

    # Normalize the array to range [0,1]
    numpy_array = numpy_array / 255.0
    
    # Get the shape of the numpy array
    nx, ny = numpy_array.shape
    
    # Define coordinates for the image
    dx = arr_width / nx
    dy = arr_height / ny
    x = np.linspace(0.5 * dx, arr_width - 0.5 * dx, nx)
    y = np.linspace(0.5 * dy, arr_height - 0.5 * dy, ny)
    
    # Create a 2D interpolation of the image data
    interp_spline = RectBivariateSpline(x, y, numpy_array)
    
    def interpolator(x):
        #print(f"interpolating at {x}")
        return interp_spline(x[0], x[1])[0, 0]

    # Create a FEniCS Function and interpolate
    class MyExpression(fe.UserExpression):
        def eval(self, value, x):
            value[0] = interpolator(x)

        def value_shape(self):
            return tuple()

    f = fe.Function(V)
    f.interpolate(MyExpression(degree=3))

    # Now f is a FEniCS Function with values interpolated from the grayscale image
    return f

