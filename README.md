# wave-from-video

Turn a video of a water reflection into a waveform.

The source footage shows light reflected off moving water as a bright band that
wiggles across a textured background — it already looks like an oscilloscope
trace. This project extracts that band from every frame and produces:

- a **spatial** waveform (the band's vertical displacement across each frame), and
- a **temporal** waveform (how the reflection oscillates over time across frames).

The primary validations are an **overlay image** (the extracted waveform drawn on
top of a reference frame) and **waveform data files** (`.npz` for matplotlib and
`.wav` for audio) that render the usual way in a Jupyter notebook.

Built with [mise](https://mise.jdx.dev/) + [uv](https://docs.astral.sh/uv/).

> Status: under active development. See `spec.md` for the full design and
> `todo.md` for progress.
