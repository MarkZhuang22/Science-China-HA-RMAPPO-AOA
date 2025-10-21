
import socket

# Make absl optional to avoid hard dependency during imports.
try:
    from absl import flags  # type: ignore
    FLAGS = flags.FLAGS
    # Provide a dummy argv to avoid absl complaining when imported
    if not FLAGS.is_parsed():
        FLAGS(["runner"])
except Exception:
    # If absl is not installed, silently continue; no flags are required here.
    pass
