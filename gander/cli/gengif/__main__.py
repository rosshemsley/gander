from typing import Optional
import struct
import os
import io

import click
from PIL import Image
import tensorboard.compat.proto.event_pb2 as event_pb2


@click.command()
@click.argument(
    "logfile",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
)
@click.option(
    "--output",
    "-o",
    default="output.gif",
    help="Name of output file.",
)
@click.option(
    "--image-width",
    default=256,
    type=int,
    help="set the width of the output gif in pixels",
)
@click.option(
    "--frame-duration-ms",
    default=500,
    type=int,
    help="Frame duration in milliseconds (will be rounded to nearest 100th of a second).",
)
@click.option(
    "--image-tag",
    default="images.generated",
    help="Writes a gif with all images that have this tag",
)
@click.option(
    "--step",
    default=100,
    type=int,
    help="only write every --step images to the output."
)
def main(logfile: str, image_tag: str, step: int, output: str, frame_duration_ms: int, image_width: int):
    images = []
    count = 0

    with open(logfile, "rb") as infile:
        event = event_pb2.Event()

        for msg in message_iterator(infile):
            event.ParseFromString(msg)

            img = extract_image_if_present(event, image_tag)
            if img is not None:
                print(f"Found image {image_tag} (found {count} so far)")
                if count % step == 0:
                    images.append(img)
                count += 1

    if len(images) == 0:
        print(f"No images found with tag {image_tag}")
        exit(1)

    resized_images = []
    for img in images:
        img = img.resize((image_width, int(img.height * image_width/img.width)))
        resized_images.append(img)

    print(f"Writing gif with {len(images)} frames")
    im = resized_images[0]
    duration = int(frame_duration_ms)
    im.save(output, save_all=True, append_images=resized_images, duration=duration, loop=0)


def extract_image_if_present(event, image_tag) -> Optional[Image.Image]:
    if event.HasField("summary"):
        for value in event.summary.value:
            if value.HasField("image") and value.tag == image_tag:
                buf = value.image.encoded_image_string
                return Image.open(io.BytesIO(buf))


def message_iterator(f):
    """
    Returns an iterator over serialized proto messages in a file-like bytestream.
    """
    while True:
        header = f.read(12)
        if len(header) != 12:
            break

        msg_size = struct.unpack("Q", header[:8])[0]

        msg = f.read(msg_size)
        if len(f.read(4)) != 4:
            raise ValueError("Malfored tensorflow logfile")

        yield msg


if __name__ == "__main__":
    main()
