import pathlib
import click
from imgcat import imgcat
import torchvision

TRUNCATE_LATENT_VECTOR = 2.0


@click.command()
@click.option(
    "--model-checkpoint",
    required=False,
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="Path to checkpoint.",
)
def main(model_checkpoint: str):
    model = GAN.load_from_checkpoint(model_checkpoint)

    latent = model.random_latent_vectors(1)
    latent = latent.clamp(min=-TRUNCATE_LATENT_VECTOR, max=TRUNCATE_LATENT_VECTOR)
    img = model(model.random_latent_vectors(16), 5)
    imgcat(img)


if __name__ == "__main__":
    main()
