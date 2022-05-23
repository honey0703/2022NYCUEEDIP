1. Create 'input/', 'output/', 'attack/' folders.
2. Put cover image and watermark image in 'input/' folder.
3. Modified the cover image and watermark image path in __main__:
    # ==================== File path setting ===================== #
    # ---- cover and watermark path -----
    input_img = 'input/lena.bmp'
    input_wm = 'input/watermark.png'
    ...
    # ============================================================ #
    The rest could remain the same as default.
4. Run the code, and it suppose to get:
    1 resize image in 'input/'
    4 attacked images in 'attack/' (include 1 resized compressed image)
    1 embedded cover image, 3 noisy watermarks and 3 post-process watermarks.




