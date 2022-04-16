import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from PIL import ImageOps
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO


# def save_module(url, save_path):
#   module = hub.KerasLayer(url)
#   model = tf.keras.Sequential(module)
#   tf.saved_model.save(model, save_path)
#
# save_module("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1", "tmp/models/inception_resnet_v2/")

#
# module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
module_handle = "tmp/models/faster_rcnn_openimages_v4_inception_resnet_v2_1/"
model = hub.load(module_handle)

print("hub.load(module_handle).signatures.keys(): ", hub.load(module_handle).signatures.keys())


# If you want to see if a model has more than one signature then you can do something like
# print(hub.load(module_handle).signatures.keys()). In your case, the models you will be using
# only have the default signature so you don't have to worry about other types.

model.signatures.keys()

print("model.signatures.keys(): ", list(model.signatures.keys()))

detector = model.signatures['default']


def download_and_resize_image(url, new_width=256, new_height=256):
  '''
  Fetches an image online, resizes it and saves it locally.

  Args:
      url (string) -- link to the image
      new_width (int) -- size in pixels used for resizing the width of the image
      new_height (int) -- size in pixels used for resizing the length of the image

  Returns:
      (string) -- path to the saved image
  '''

  # create a temporary file ending with ".jpg"
  _, filename = tempfile.mkstemp(suffix=".jpg")

  # opens the given URL
  response = urlopen(url)

  # reads the image fetched from the URL
  image_data = response.read()

  # puts the image data in memory buffer
  image_data = BytesIO(image_data)

  # opens the image
  pil_image = Image.open(image_data)

  # resizes the image. will crop if aspect ratio is different.
  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)

  # converts to the RGB colorspace
  pil_image_rgb = pil_image.convert("RGB")

  # saves the image to the temporary file created earlier
  pil_image_rgb.save(filename, format="JPEG", quality=90)

  print("Image downloaded to %s." % filename)

  return filename


# You can choose a different URL that points to an image of your choice
image_url = "https://upload.wikimedia.org/wikipedia/commons/f/fb/20130807_dublin014.JPG"

# download the image and use the original height and width
downloaded_image_path = download_and_resize_image(image_url, 3872, 2592)


def load_img(path):
  '''
  Loads a JPEG image and converts it to a tensor.

  Args:
      path (string) -- path to a locally saved JPEG image

  Returns:
      (tensor) -- an image tensor
  '''

  # read the file
  img = tf.io.read_file(path)

  # convert to a tensor
  img = tf.image.decode_jpeg(img, channels=3)

  return img


def run_detector(detector, path):
  '''
  Runs inference on a local file using an object detection model.

  Args:
      detector (model) -- an object detection model loaded from TF Hub
      path (string) -- path to an image saved locally
  '''

  # load an image tensor from a local file path
  img = load_img(path)

  # add a batch dimension in front of the tensor
  converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

  # run inference using the model
  result = detector(converted_img)

  # save the results in a dictionary
  result = {key: value.numpy() for key, value in result.items()}

  # print results
  print("Found %d objects." % len(result["detection_scores"]))

  print(result["detection_scores"])
  print(result["detection_class_entities"])
  print(result["detection_boxes"])

run_detector(detector, downloaded_image_path)