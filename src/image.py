from PIL import Image, ImageFilter


def srcnn_img_preprocess(img: Image.Image, scale: int = 2):
    new_height = (img.height // scale) * scale
    new_width = (img.width // scale) * scale
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

    lr_img = img.filter(ImageFilter.GaussianBlur(radius=2))
    lr_img = lr_img.resize(
        (lr_img.width // scale, lr_img.height // scale),
        Image.Resampling.BICUBIC,
    )
    lr_img = lr_img.resize(
        (lr_img.width * scale, lr_img.height * scale),
        Image.Resampling.BICUBIC,
    )

    return lr_img, img
